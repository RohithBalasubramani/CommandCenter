from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VoiceSessionViewSet, TranscriptViewSet, PersonaPlexConfigViewSet

router = DefaultRouter()
router.register(r"sessions", VoiceSessionViewSet, basename="voice-session")
router.register(r"transcripts", TranscriptViewSet, basename="transcript")
router.register(r"config", PersonaPlexConfigViewSet, basename="personaplex-config")

urlpatterns = [
    path("", include(router.urls)),
]
