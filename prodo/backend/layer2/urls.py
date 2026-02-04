from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    RAGPipelineViewSet,
    RAGQueryViewSet,
    industrial_rag_query,
    industrial_rag_health,
    orchestrate,
    get_filler,
    proactive_trigger,
)

router = DefaultRouter()
router.register(r"pipelines", RAGPipelineViewSet, basename="rag-pipeline")
router.register(r"queries", RAGQueryViewSet, basename="rag-query")

urlpatterns = [
    path("", include(router.urls)),
    # Main orchestration endpoint (Layer 2 brain)
    path("orchestrate/", orchestrate, name="layer2-orchestrate"),
    path("filler/", get_filler, name="layer2-filler"),
    path("proactive/", proactive_trigger, name="layer2-proactive"),
    # RAG pipeline endpoints
    path("rag/industrial/", industrial_rag_query, name="industrial-rag-query"),
    path("rag/industrial/health/", industrial_rag_health, name="industrial-rag-health"),
]
