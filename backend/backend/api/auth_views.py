from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth.models import User
from .models import Academy, PlayerProfile
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('email')  # Using email as username for simplicity
        password = request.data.get('password')
        name = request.data.get('name', 'User')
        role = request.data.get('role', 'COACH') # Default to coach for backward compatibility

        if not username or not password:
            return Response({'error': 'Email and password required'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({'error': 'User already exists'}, status=status.HTTP_400_BAD_REQUEST)

        # Create user
        user = User.objects.create_user(username=username, email=username, password=password)

        # Assign to appropriate profile
        is_verified = None
        if role == 'LEARNER':
            batting_hand = request.data.get('batting_hand', 'Right')
            if batting_hand not in ('Right', 'Left'):
                batting_hand = 'Right'
            playing_level = request.data.get('playing_level') or 'club'
            PlayerProfile.objects.create(
                user=user, name=name, batting_hand=batting_hand, playing_level=playing_level
            )
        else:
            # ISSUE-007 / FR-AUTH-003: the account exists immediately (so the
            # coach can log in and see their pending status) but starts
            # unverified -- Academy.is_verified defaults to False, and
            # coach-only actions in views.py stay locked until an admin
            # flips it via Django admin.
            academy = Academy.objects.create(user=user, academy_name=name)
            is_verified = academy.is_verified

        # Generate tokens
        refresh = RefreshToken.for_user(user)
        # Custom claim for role
        refresh['role'] = role

        user_payload = {
            'id': user.id,
            'email': user.email,
            'role': role,
            'name': name
        }
        if is_verified is not None:
            user_payload['is_verified'] = is_verified

        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': user_payload
        }, status=status.HTTP_201_CREATED)

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        
        # Determine role and profile data
        role = 'UNKNOWN'
        name = 'Unknown'
        is_verified = None

        if hasattr(user, 'academy'):
            role = 'COACH'
            name = user.academy.academy_name
            is_verified = user.academy.is_verified  # ISSUE-007 / FR-AUTH-003
        elif hasattr(user, 'playerprofile'):
            role = 'LEARNER'
            name = user.playerprofile.name

        payload = {
            'id': user.id,
            'email': user.email,
            'role': role,
            'name': name
        }
        if is_verified is not None:
            payload['is_verified'] = is_verified

        return Response(payload)
