import os
import uuid
import logging
from django.conf import settings
from rest_framework import viewsets, views
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import PermissionDenied
from .models import PlayerProfile, AnalysisSession, Academy
from .serializers import AnalysisSessionSerializer, PlayerProfileSerializer
from .ml_service import run_advanced_inference

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

    def perform_update(self, serializer):
        user = self.request.user
        if not hasattr(user, 'academy'):
            raise PermissionDenied("Only a coach can override a session's scores.")
        serializer.save()

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
            player_id = request.data.get('player_id')
            if not player_id:
                return Response({"error": "Coach must provide player_id."}, status=400)
            player, _ = PlayerProfile.objects.get_or_create(
                id=player_id, academy=user.academy, defaults={'name': 'Guest Player', 'batting_hand': 'Right'}
            )
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
        except Exception as e:
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
            balance_score=scores_data.get('balance_score', 0),
            power_score=scores_data.get('power_score', 0),
            technique_score=scores_data.get('technique_score', 0),
            defence_score=scores_data.get('defence_score', 0)
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
        
        academy = user.academy
        players = PlayerProfile.objects.filter(academy=academy)
        sessions = AnalysisSession.objects.filter(player__in=players).order_by('-date_analyzed')
        
        return Response({
            "academy": {"name": academy.academy_name},
            "players": PlayerProfileSerializer(players, many=True).data,
            "review_queue": AnalysisSessionSerializer(sessions, many=True).data
        })