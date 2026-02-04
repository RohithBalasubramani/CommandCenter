from rest_framework import serializers
from .models import VoiceSession, Transcript, PersonaPlexConfig


class TranscriptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transcript
        fields = ["id", "session", "role", "text", "is_final", "created_at"]
        read_only_fields = ["id", "created_at"]


class VoiceSessionSerializer(serializers.ModelSerializer):
    transcripts = TranscriptSerializer(many=True, read_only=True)

    class Meta:
        model = VoiceSession
        fields = ["id", "started_at", "ended_at", "status", "transcripts"]
        read_only_fields = ["id", "started_at"]


class PersonaPlexConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonaPlexConfig
        fields = ["id", "server_url", "model", "voice", "always_on", "updated_at"]
        read_only_fields = ["id", "updated_at"]
