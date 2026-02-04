from rest_framework import serializers
from .models import WidgetRating, WidgetFeedback


class WidgetRatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = WidgetRating
        fields = [
            "id", "entry_id", "rating", "tags", "notes",
            "rated_at", "device_id", "created_at", "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class WidgetFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = WidgetFeedback
        fields = [
            "id", "scenario", "variant", "size_mode",
            "feedback_type", "data", "device_id", "created_at",
        ]
        read_only_fields = ["id", "created_at"]
