"""
Action models for Command Center Pipeline v2.

Supports: reminders, messages, device commands, tasks, and an audit log.
"""

import uuid
from django.db import models


class Reminder(models.Model):
    """Scheduled reminder triggered by voice command."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.TextField()
    trigger_time = models.DateTimeField()
    recurring = models.CharField(max_length=20, blank=True, default="")  # daily, weekly, etc.
    status = models.CharField(
        max_length=20,
        default="pending",
        choices=[("pending", "Pending"), ("triggered", "Triggered"), ("dismissed", "Dismissed")],
    )
    entity = models.CharField(max_length=200, blank=True, default="")  # related equipment/person
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-trigger_time"]

    def __str__(self):
        return f"Reminder({self.status}): {self.message[:50]}"


class Message(models.Model):
    """Message sent via voice command."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recipient = models.CharField(max_length=200)  # team name, person, role
    content = models.TextField()
    channel = models.CharField(
        max_length=20,
        default="internal",
        choices=[("internal", "Internal"), ("sms", "SMS"), ("email", "Email")],
    )
    status = models.CharField(
        max_length=20,
        default="queued",
        choices=[("queued", "Queued"), ("sent", "Sent"), ("failed", "Failed")],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Message to {self.recipient}: {self.content[:50]}"


class DeviceCommand(models.Model):
    """Device control command issued via voice."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    device_type = models.CharField(max_length=50)  # pump, motor, chiller, etc.
    device_name = models.CharField(max_length=200)
    command = models.CharField(max_length=50)  # start, stop, set_parameter
    parameters = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=20,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("confirmed", "Confirmed"),
            ("executed", "Executed"),
            ("failed", "Failed"),
            ("rejected", "Rejected"),
        ],
    )
    result = models.TextField(blank=True, default="")
    requires_confirmation = models.BooleanField(default=True)
    confirmed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    executed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Command({self.status}): {self.command} {self.device_name}"


class ActionLog(models.Model):
    """Audit trail for all actions executed by the system."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    action_type = models.CharField(max_length=20)  # reminder, message, command, task
    action_id = models.UUIDField()
    transcript = models.TextField()
    intent_json = models.JSONField(default=dict)
    result = models.CharField(
        max_length=20,
        default="success",
        choices=[("success", "Success"), ("failed", "Failed"), ("pending_confirmation", "Pending Confirmation")],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"ActionLog({self.action_type}/{self.result}): {self.transcript[:50]}"
