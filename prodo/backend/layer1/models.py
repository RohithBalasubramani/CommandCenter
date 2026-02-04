import uuid
from django.db import models


class VoiceSession(models.Model):
    """A voice conversation session between user and PersonaPlex."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ("active", "Active"),
            ("completed", "Completed"),
            ("error", "Error"),
        ],
        default="active",
    )

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"VoiceSession {self.id} ({self.status})"


class Transcript(models.Model):
    """A single transcript entry (user or assistant turn)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        VoiceSession, on_delete=models.CASCADE, related_name="transcripts"
    )
    role = models.CharField(
        max_length=10,
        choices=[("user", "User"), ("assistant", "Assistant")],
    )
    text = models.TextField()
    is_final = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.text[:50]}"


class PersonaPlexConfig(models.Model):
    """PersonaPlex server configuration (singleton-ish)."""
    server_url = models.URLField(default="http://localhost:8090")
    model = models.CharField(max_length=100, default="personaplex-7b-v1")
    voice = models.CharField(max_length=20, default="NATF0")
    always_on = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "PersonaPlex Config"

    def __str__(self):
        return f"PersonaPlex ({self.model}, {self.voice})"
