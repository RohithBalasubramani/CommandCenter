from django.urls import path
from actions import views

urlpatterns = [
    path("reminders/", views.reminders_view, name="actions-reminders"),
    path("messages/", views.messages_view, name="actions-messages"),
    path("commands/", views.commands_view, name="actions-commands"),
    path("commands/<uuid:command_id>/confirm/", views.confirm_command_view, name="actions-confirm-command"),
    path("log/", views.action_log_view, name="actions-log"),
]
