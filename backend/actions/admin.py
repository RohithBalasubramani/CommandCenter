from django.contrib import admin
from actions.models import Reminder, Message, DeviceCommand, ActionLog


@admin.register(Reminder)
class ReminderAdmin(admin.ModelAdmin):
    list_display = ["id", "message", "trigger_time", "status", "entity", "created_at"]
    list_filter = ["status"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["id", "recipient", "content", "channel", "status", "created_at"]
    list_filter = ["status", "channel"]


@admin.register(DeviceCommand)
class DeviceCommandAdmin(admin.ModelAdmin):
    list_display = ["id", "device_name", "command", "status", "confirmed", "created_at"]
    list_filter = ["status", "device_type"]


@admin.register(ActionLog)
class ActionLogAdmin(admin.ModelAdmin):
    list_display = ["id", "action_type", "result", "transcript", "created_at"]
    list_filter = ["action_type", "result"]
