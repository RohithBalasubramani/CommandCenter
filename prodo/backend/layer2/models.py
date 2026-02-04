import uuid
from django.db import models


class RAGPipeline(models.Model):
    """Registry of available RAG pipelines (Layer 2B)."""
    DOMAIN_CHOICES = [
        ("industrial", "Industrial"),
        ("supply", "Supply Chain"),
        ("people", "People"),
        ("tasks", "Tasks"),
        ("alerts", "Alerts"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    domain = models.CharField(max_length=20, choices=DOMAIN_CHOICES, unique=True)
    enabled = models.BooleanField(default=True)
    priority = models.IntegerField(default=1)
    endpoint_url = models.URLField(
        help_text="Backend endpoint for this RAG pipeline's data source"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["priority"]

    def __str__(self):
        return f"RAG: {self.domain} ({'enabled' if self.enabled else 'disabled'})"


class RAGQuery(models.Model):
    """Log of RAG queries executed by the orchestrator."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        RAGPipeline, on_delete=models.CASCADE, related_name="queries"
    )
    transcript_id = models.UUIDField(
        help_text="ID of the transcript entry that triggered this query"
    )
    query_text = models.TextField()
    intent = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "RAG Queries"

    def __str__(self):
        return f"Query to {self.pipeline.domain}: {self.query_text[:50]}"


class RAGResult(models.Model):
    """Results from RAG pipeline execution."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query = models.OneToOneField(
        RAGQuery, on_delete=models.CASCADE, related_name="result"
    )
    raw_data = models.JSONField(default=dict)
    error = models.TextField(null=True, blank=True)
    execution_time_ms = models.IntegerField(
        help_text="Query execution time in milliseconds"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        status = "error" if self.error else "success"
        return f"Result ({status}): {self.query.pipeline.domain} in {self.execution_time_ms}ms"
