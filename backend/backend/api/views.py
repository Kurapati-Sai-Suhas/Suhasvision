import os
import cv2
import json
import uuid
import re
import base64
from django.conf import settings
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import PlayerProfile, AnalysisSession
from .serializers import AnalysisSessionSerializer
from .ml_service import run_advanced_inference

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class AnalysisSessionViewSet(viewsets.ModelViewSet):
    queryset = AnalysisSession.objects.all()
    serializer_class = AnalysisSessionSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'])
    def analyze_stance(self, request):
        player_id = request.data.get('player_id')
        video_file = request.FILES.get('video')
        
        if not video_file:
            return Response({"error": "No video file provided."}, status=400)
            
        if not player_id:
            return Response({"error": "No player_id provided."}, status=400)
            
        # Associate with the logged-in coach's academy
        user = request.user
        try:
            academy = user.academy
        except Exception:
            return Response({"error": "No academy profile found for this user."}, status=400)
            
        # Get or create the player for this specific academy
        try:
            player, created = PlayerProfile.objects.get_or_create(
                id=player_id,
                academy=academy,
                defaults={'name': 'Guest Player', 'batting_hand': 'Right'}
            )
        except Exception as e:
            return Response({"error": f"Database error finding player: {str(e)}"}, status=500)

        # 1. Save video temporarily to disk
        video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        unique_filename = f"{uuid.uuid4()}_{video_file.name}"
        video_path = os.path.join(video_dir, unique_filename)
        
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
                
        # 2. Run Custom Keras/MediaPipe Inference
        try:
            scores_data = run_advanced_inference(video_path)
        except Exception as e:
            return Response({"error": f"ML Inference Error: {str(e)}"}, status=500)
        
        # 3. Create Session with real AI data
        new_session = AnalysisSession.objects.create(
            player=player,
            overall_score=scores_data.get('overall_score', 0),
            primary_weakness=scores_data.get('primary_weakness', "Unknown"),
            primary_strength=scores_data.get('primary_strength', "Unknown"),
            balance_score=scores_data.get('balance_score', 0),
            power_score=scores_data.get('power_score', 0),
            technique_score=scores_data.get('technique_score', 0),
            defence_score=scores_data.get('defence_score', 0)
        )

        # For the frontend response, we don't have the 4 exact extracted frames since 
        # the Keras model runs 7 evenly spaced frames and operates on the tensor directly.
        # We can just return a generic list or empty list if the frontend expects frame_urls.
        
        return Response({
            "status": "Success", 
            "session_id": new_session.id,
            "scores": scores_data,
            "frame_urls": [] # Removed since Llama's timeline extraction is gone
        })