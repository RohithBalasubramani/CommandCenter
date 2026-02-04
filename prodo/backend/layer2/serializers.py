from rest_framework import serializers
from .models import RAGPipeline, RAGQuery, RAGResult


class RAGResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGResult
        fields = ["id", "raw_data", "error", "execution_time_ms", "created_at"]
        read_only_fields = ["id", "created_at"]


class RAGQuerySerializer(serializers.ModelSerializer):
    result = RAGResultSerializer(read_only=True)

    class Meta:
        model = RAGQuery
        fields = ["id", "pipeline", "transcript_id", "query_text", "intent", "result", "created_at"]
        read_only_fields = ["id", "created_at"]


class RAGPipelineSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGPipeline
        fields = ["id", "domain", "enabled", "priority", "endpoint_url", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]
