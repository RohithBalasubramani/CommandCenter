from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import WidgetRating, WidgetFeedback
from .serializers import WidgetRatingSerializer, WidgetFeedbackSerializer


# ── Widget Ratings ──


@api_view(["GET", "POST"])
def ratings_list(request):
    """
    GET  — return all ratings (keyed by entry_id for easy frontend merge).
    POST — upsert a single rating (entry_id + device_id = unique key).
    """
    if request.method == "GET":
        qs = WidgetRating.objects.all()
        serializer = WidgetRatingSerializer(qs, many=True)
        return Response(serializer.data)

    # POST — upsert
    entry_id = request.data.get("entry_id")
    device_id = request.data.get("device_id", "")
    if not entry_id:
        return Response(
            {"error": "entry_id is required"}, status=status.HTTP_400_BAD_REQUEST
        )

    serializer = WidgetRatingSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    # Upsert: update if exists, create otherwise
    obj, created = WidgetRating.objects.update_or_create(
        entry_id=entry_id,
        device_id=device_id,
        defaults={
            "rating": serializer.validated_data["rating"],
            "tags": serializer.validated_data.get("tags", []),
            "notes": serializer.validated_data.get("notes", ""),
            "rated_at": serializer.validated_data["rated_at"],
        },
    )
    out = WidgetRatingSerializer(obj)
    return Response(
        out.data,
        status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
    )


@api_view(["POST"])
def ratings_bulk(request):
    """
    POST — bulk upsert ratings.
    Body: { "ratings": { entry_id: { rating, tags, notes, rated_at } }, "device_id": "..." }
    Used by frontend on initial sync to push all localStorage ratings at once.
    """
    ratings_map = request.data.get("ratings", {})
    device_id = request.data.get("device_id", "")

    if not isinstance(ratings_map, dict):
        return Response(
            {"error": "ratings must be an object keyed by entry_id"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    created_count = 0
    updated_count = 0

    for entry_id, payload in ratings_map.items():
        _, created = WidgetRating.objects.update_or_create(
            entry_id=entry_id,
            device_id=device_id,
            defaults={
                "rating": payload.get("rating", "up"),
                "tags": payload.get("tags", []),
                "notes": payload.get("notes", ""),
                "rated_at": payload.get("rated_at"),
            },
        )
        if created:
            created_count += 1
        else:
            updated_count += 1

    return Response(
        {"created": created_count, "updated": updated_count, "total": len(ratings_map)}
    )


# ── Widget Feedback ──


@api_view(["GET", "POST"])
def feedback_list(request):
    """
    GET  — return all feedback entries.
    POST — create a new feedback entry.
    """
    if request.method == "GET":
        qs = WidgetFeedback.objects.all()
        serializer = WidgetFeedbackSerializer(qs, many=True)
        return Response(serializer.data)

    serializer = WidgetFeedbackSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    serializer.save()
    return Response(serializer.data, status=status.HTTP_201_CREATED)
