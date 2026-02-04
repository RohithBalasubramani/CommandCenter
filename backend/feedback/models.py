import uuid
from django.db import models


class WidgetRating(models.Model):
    """A single rating for a widget entry from the exhaustive simulation."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    entry_id = models.CharField(max_length=255, db_index=True)
    rating = models.CharField(max_length=10, choices=[("up", "Up"), ("down", "Down")])
    tags = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True, default="")
    rated_at = models.DateTimeField()
    device_id = models.CharField(
        max_length=255, blank=True, default="",
        help_text="Browser fingerprint or device identifier",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-rated_at"]
        # One rating per entry per device — latest wins
        constraints = [
            models.UniqueConstraint(
                fields=["entry_id", "device_id"],
                name="unique_rating_per_device",
            )
        ]

    def __str__(self):
        return f"{self.entry_id} → {self.rating} ({self.device_id})"


class WidgetFeedback(models.Model):
    """Size/issue feedback for a specific widget variant from the gallery."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    scenario = models.CharField(max_length=255)
    variant = models.CharField(max_length=255)
    size_mode = models.CharField(max_length=50, blank=True, default="")
    feedback_type = models.CharField(
        max_length=50,
        choices=[("size", "Size Adjustment"), ("issue", "Issue Report")],
        default="issue",
    )
    data = models.JSONField(default=dict, help_text="Full feedback payload")
    device_id = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.scenario}/{self.variant} — {self.feedback_type}"
