from rest_framework import serializers
from .models import Academy, PlayerProfile, AnalysisSession

class PlayerProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlayerProfile
        fields = '__all__'  # <-- Changed from - to =

class AnalysisSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisSession # <-- Changed from - to =
        fields = '__all__'      # <-- Changed from - to =