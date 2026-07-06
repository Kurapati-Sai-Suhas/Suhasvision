from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AnalysisSessionViewSet
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .auth_views import RegisterView, UserProfileView

router = DefaultRouter()
router.register(r'sessions', AnalysisSessionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/login/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/register/', RegisterView.as_view(), name='auth_register'),
    path('auth/me/', UserProfileView.as_view(), name='auth_me'),
]
