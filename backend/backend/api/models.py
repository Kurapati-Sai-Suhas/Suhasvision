# Create your models here.
from django.db import models
from django.contrib.auth.models import User

# 1. The Coach/Academy Account (Links to Django's built-in User auth)
class Academy(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    academy_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.academy_name

# 2. The Player Profile (Learners who can log in)
class PlayerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    academy = models.ForeignKey(Academy, related_name='players', on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=100)
    batting_hand = models.CharField(max_length=20, choices=[('Right', 'Right'), ('Left', 'Left')])
    playing_level = models.CharField(max_length=50)
    archetype = models.CharField(max_length=50, blank=True, null=True) # e.g., "Classical Technician"
    
    # Gamification
    total_points = models.IntegerField(default=0)
    current_streak = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.name}"

# 3. The Video Analysis Session (A player can have multiple sessions for comparison)
class AnalysisSession(models.Model):
    player = models.ForeignKey(PlayerProfile, related_name='sessions', on_delete=models.CASCADE)
    video_url = models.URLField(max_length=500, blank=True, null=True) # Will point to your Azure Blob!
    date_analyzed = models.DateTimeField(auto_now_add=True)
    
    # The 4-Factor AI Output
    overall_score = models.IntegerField(default=0)
    primary_weakness = models.TextField(blank=True, null=True)
    primary_strength = models.TextField(blank=True, null=True)
    thing_to_change = models.TextField(blank=True, null=True)
    bonus_insight = models.TextField(blank=True, null=True)
    
    # The Metrics
    balance_score = models.IntegerField(default=0)
    power_score = models.IntegerField(default=0)
    technique_score = models.IntegerField(default=0)
    defence_score = models.IntegerField(default=0)

    def __str__(self):
        return f"Session: {self.player.name} - {self.date_analyzed.strftime('%Y-%m-%d')}"

# 4. Coach Override Log (Traceability for AI adjustments)
class CoachOverrideLog(models.Model):
    session = models.ForeignKey(AnalysisSession, related_name='overrides', on_delete=models.CASCADE)
    coach = models.ForeignKey(Academy, on_delete=models.CASCADE)
    metric_changed = models.CharField(max_length=50) # e.g., 'power_score'
    old_value = models.CharField(max_length=255)
    new_value = models.CharField(max_length=255)
    reason = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Override by {self.coach.academy_name} on {self.session.id}"