from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/layer1/", include("layer1.urls")),
    path("api/layer2/", include("layer2.urls")),
    path("api/actions/", include("actions.urls")),
    path("api/", include("feedback.urls")),
]
