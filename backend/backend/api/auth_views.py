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
        if role == 'LEARNER':
            PlayerProfile.objects.create(user=user, name=name, batting_hand='Right', playing_level='club')
        else:
            Academy.objects.create(user=user, academy_name=name)

        # Generate tokens
        refresh = RefreshToken.for_user(user)
        # Custom claim for role
        refresh['role'] = role

        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'id': user.id,
                'email': user.email,
                'role': role,
                'name': name
            }
        }, status=status.HTTP_201_CREATED)

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        
        # Determine role and profile data
        role = 'UNKNOWN'
        name = 'Unknown'
        
        if hasattr(user, 'academy'):
            role = 'COACH'
            name = user.academy.academy_name
        elif hasattr(user, 'playerprofile'):
            role = 'LEARNER'
            name = user.playerprofile.name

        return Response({
            'id': user.id,
            'email': user.email,
            'role': role,
            'name': name
        })
