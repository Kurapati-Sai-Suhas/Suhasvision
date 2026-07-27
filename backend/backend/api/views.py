import os
import uuid
import logging
from django.conf import settings
from rest_framework import viewsets, views
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import PermissionDenied
from .models import PlayerProfile, AnalysisSession, Academy, CoachOverrideLog
from .serializers import AnalysisSessionSerializer, PlayerProfileSerializer
from .ml_service import run_advanced_inference, InvalidUploadError

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"}
MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200MB

class AnalysisSessionViewSet(viewsets.ModelViewSet):
    queryset = AnalysisSession.objects.all()
    serializer_class = AnalysisSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, 'playerprofile'):
            return AnalysisSession.objects.filter(player=user.playerprofile).order_by('-date_analyzed')
        elif hasattr(user, 'academy'):
            return AnalysisSession.objects.filter(player__academy=user.academy).order_by('-date_analyzed')
        return AnalysisSession.objects.none()

    def create(self, request, *args, **kwargs):
        # Sessions may only be created via analyze_stance, which enforces player
        # scoping and runs the ML pipeline. A bare POST here would let any
        # authenticated user write arbitrary scores against any player id.
        return Response(
            {"error": "Direct session creation is not allowed. Use /sessions/analyze_stance/."},
            status=405,
        )

    OVERRIDABLE_METRICS = ['balance_score', 'power_score', 'technique_score', 'defence_score']

    def perform_update(self, serializer):
        user = self.request.user
        if not hasattr(user, 'academy'):
            raise PermissionDenied("Only a coach can override a session's scores.")
        if not user.academy.is_verified:
            # ISSUE-007 / FR-AUTH-003: an unverified coach account exists but
            # shouldn't be able to write scores yet.
            raise PermissionDenied(
                "Your coach account is pending admin verification and cannot override scores yet."
            )

        session = serializer.instance
        before = {m: getattr(session, m) for m in self.OVERRIDABLE_METRICS}
        updated = serializer.save()

        # Real traceability for the CoachOverrideLog model (SRS-documented,
        # previously never written to) -- log every metric a coach actually
        # changed, not just that an update happened.
        for metric in self.OVERRIDABLE_METRICS:
            old_value, new_value = before[metric], getattr(updated, metric)
            if old_value != new_value:
                CoachOverrideLog.objects.create(
                    session=updated,
                    coach=user.academy,
                    metric_changed=metric,
                    old_value=str(old_value),
                    new_value=str(new_value),
                    reason=self.request.data.get('bonus_insight') or '',
                )

    @action(detail=False, methods=['post'])
    def analyze_stance(self, request):
        user = request.user
        video_file = request.FILES.get('video')
        title = request.data.get('title', 'Practice Session')
        
        if not video_file:
            return Response({"error": "No video file provided."}, status=400)

        if video_file.size > MAX_VIDEO_BYTES:
            return Response({"error": "Video exceeds the 200MB upload limit."}, status=400)

        if video_file.content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
            return Response(
                {"error": "Unsupported file type. Please upload an MP4, MOV, WEBM, or AVI video."},
                status=400,
            )

        # Determine player context
        player = None
        if hasattr(user, 'playerprofile'):
            player = user.playerprofile
        elif hasattr(user, 'academy'):
            if not user.academy.is_verified:
                # ISSUE-007 / FR-AUTH-003: gate coach privileges behind admin
                # verification (Academy.is_verified) rather than granting them
                # immediately at signup.
                return Response(
                    {"error": "Your coach account is pending admin verification. "
                              "You'll be able to analyze videos for players once an admin approves it."},
                    status=403,
                )
            player_id = request.data.get('player_id')
            if not player_id:
                return Response({"error": "Coach must provide player_id."}, status=400)
            # Look up, never create: the old get_or_create wrote a row with a
            # CLIENT-SUPPLIED primary key, which 500'd with an IntegrityError
            # whenever that id already existed in another academy (audit H7).
            # ValueError covers a non-integer player_id, which the pk lookup
            # would otherwise raise on.
            try:
                player = PlayerProfile.objects.get(id=player_id, academy=user.academy)
            except (PlayerProfile.DoesNotExist, ValueError):
                return Response({"error": "No player with that id in your academy."}, status=404)
        else:
            return Response({"error": "User role invalid for upload."}, status=403)

        # 1. Save video temporarily to disk
        video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        unique_filename = f"{uuid.uuid4()}_{video_file.name}"
        video_path = os.path.join(video_dir, unique_filename)
        
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
                
        # 2. Run Custom Keras/MediaPipe Inference
        # The uploaded video only ever needs to exist on disk for the duration of
        # this call (zero-storage requirement) — always clean it up afterwards,
        # whether inference succeeds or fails.
        try:
            scores_data = run_advanced_inference(video_path)
        except InvalidUploadError as e:
            # Upload-contract rejection (Milestone 3, Option B): the clip
            # itself is the problem (too long, unreadable, no trackable
            # pose), and the message says exactly what to fix — a 400, not
            # a 500 pretending the server broke.
            return Response({"error": str(e)}, status=400)
        except Exception:
            logger.exception("ML inference failed for %s", unique_filename)
            return Response(
                {"error": "We couldn't analyze that video. Please try a clearer, well-lit clip."},
                status=500,
            )
        finally:
            try:
                os.remove(video_path)
            except OSError:
                logger.warning("Could not remove temp video %s", video_path)

        # 3. Create Session with real AI data
        new_session = AnalysisSession.objects.create(
            player=player,
            title=title,
            status="COMPLETED",
            overall_score=scores_data.get('overall_score', 0),
            primary_weakness=scores_data.get('primary_weakness', "Unknown"),
            primary_strength=scores_data.get('primary_strength', "Unknown"),
            thing_to_change=scores_data.get('recommended_drill'),
            balance_score=scores_data.get('balance_score', 0),
            power_score=scores_data.get('power_score', 0),
            technique_score=scores_data.get('technique_score', 0),
            defence_score=scores_data.get('defence_score', 0),
            confidence_variance=scores_data.get('confidence_variance'),
            attribution_drivers=scores_data.get('attribution_drivers'),
            is_fallback=scores_data.get('is_fallback', False),
        )
        
        return Response({
            "status": "Success", 
            "session_id": new_session.id,
            "session": AnalysisSessionSerializer(new_session).data
        })

class LearnerDashboardView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if not hasattr(user, 'playerprofile'):
            return Response({"error": "User is not a learner"}, status=403)
        
        profile = user.playerprofile
        sessions = AnalysisSession.objects.filter(player=profile).order_by('-date_analyzed')
        
        return Response({
            "profile": PlayerProfileSerializer(profile).data,
            "sessions": AnalysisSessionSerializer(sessions, many=True).data
        })

class CoachDashboardView(views.APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if not hasattr(user, 'academy'):
            return Response({"error": "User is not a coach"}, status=403)
        if not user.academy.is_verified:
            # ISSUE-007 / FR-AUTH-003
            return Response(
                {"error": "Your coach account is pending admin verification."},
                status=403,
            )

        academy = user.academy
        players = PlayerProfile.objects.filter(academy=academy)
        sessions = AnalysisSession.objects.filter(player__in=players).order_by('-date_analyzed')
        
        return Response({
            "academy": {"name": academy.academy_name},
            "players": PlayerProfileSerializer(players, many=True).data,
            "review_queue": AnalysisSessionSerializer(sessions, many=True).data
        })