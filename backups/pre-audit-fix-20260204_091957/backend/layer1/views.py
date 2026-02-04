from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone

from .models import VoiceSession, Transcript, PersonaPlexConfig
from .serializers import (
    VoiceSessionSerializer,
    TranscriptSerializer,
    PersonaPlexConfigSerializer,
)


class VoiceSessionViewSet(viewsets.ModelViewSet):
    """
    Layer 1 — Voice session management.

    Endpoints:
      GET    /api/layer1/sessions/           — List all sessions
      POST   /api/layer1/sessions/           — Start a new session
      GET    /api/layer1/sessions/{id}/       — Get session with transcripts
      PATCH  /api/layer1/sessions/{id}/       — Update session (e.g., end it)
      POST   /api/layer1/sessions/{id}/end/   — End a session
    """
    queryset = VoiceSession.objects.all()
    serializer_class = VoiceSessionSerializer

    @action(detail=True, methods=["post"])
    def end(self, request, pk=None):
        session = self.get_object()
        session.status = "completed"
        session.ended_at = timezone.now()
        session.save()
        return Response(VoiceSessionSerializer(session).data)


class TranscriptViewSet(viewsets.ModelViewSet):
    """
    Layer 1 — Transcript entries.

    Endpoints:
      GET    /api/layer1/transcripts/         — List all transcripts
      POST   /api/layer1/transcripts/         — Add a transcript entry
      GET    /api/layer1/transcripts/{id}/     — Get single transcript
    """
    queryset = Transcript.objects.all()
    serializer_class = TranscriptSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        session_id = self.request.query_params.get("session")
        if session_id:
            qs = qs.filter(session_id=session_id)
        return qs


class PersonaPlexConfigViewSet(viewsets.ModelViewSet):
    """
    Layer 1 — PersonaPlex configuration.

    Endpoints:
      GET    /api/layer1/config/              — Get current config
      POST   /api/layer1/config/              — Create/update config
    """
    queryset = PersonaPlexConfig.objects.all()
    serializer_class = PersonaPlexConfigSerializer

    @action(detail=False, methods=["get"])
    def current(self, request):
        config = PersonaPlexConfig.objects.first()
        if not config:
            config = PersonaPlexConfig.objects.create()
        return Response(PersonaPlexConfigSerializer(config).data)
