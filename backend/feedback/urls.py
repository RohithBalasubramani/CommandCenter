from django.urls import path
from . import views

urlpatterns = [
    path("ratings/", views.ratings_list, name="ratings-list"),
    path("ratings/bulk/", views.ratings_bulk, name="ratings-bulk"),
    path("feedback/", views.feedback_list, name="feedback-list"),
]
