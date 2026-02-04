"""
Action API views for Command Center Pipeline v2.

Endpoints:
    GET  /api/actions/reminders/              List pending reminders
    POST /api/actions/reminders/              Create reminder
    GET  /api/actions/messages/               List recent messages
    GET  /api/actions/commands/               List device commands
    POST /api/actions/commands/<id>/confirm/  Confirm a pending command
    GET  /api/actions/log/                    Audit trail
"""

import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from actions.models import Reminder, Message, DeviceCommand, ActionLog


@csrf_exempt
@require_http_methods(["GET", "POST"])
def reminders_view(request):
    """List or create reminders."""
    if request.method == "GET":
        status = request.GET.get("status", "pending")
        qs = Reminder.objects.all()
        if status != "all":
            qs = qs.filter(status=status)
        reminders = [
            {
                "id": str(r.id),
                "message": r.message,
                "trigger_time": r.trigger_time.isoformat(),
                "recurring": r.recurring,
                "status": r.status,
                "entity": r.entity,
                "created_at": r.created_at.isoformat(),
            }
            for r in qs[:50]
        ]
        return JsonResponse({"reminders": reminders})

    # POST: create reminder
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    from django.utils.dateparse import parse_datetime
    trigger_time = parse_datetime(data.get("trigger_time", ""))
    if not trigger_time:
        return JsonResponse({"error": "trigger_time is required (ISO format)"}, status=400)

    reminder = Reminder.objects.create(
        message=data.get("message", ""),
        trigger_time=trigger_time,
        recurring=data.get("recurring", ""),
        entity=data.get("entity", ""),
    )
    return JsonResponse({"id": str(reminder.id), "status": "created"}, status=201)


@csrf_exempt
@require_http_methods(["GET"])
def messages_view(request):
    """List recent messages."""
    status = request.GET.get("status", "all")
    qs = Message.objects.all()
    if status != "all":
        qs = qs.filter(status=status)
    messages = [
        {
            "id": str(m.id),
            "recipient": m.recipient,
            "content": m.content,
            "channel": m.channel,
            "status": m.status,
            "created_at": m.created_at.isoformat(),
        }
        for m in qs[:50]
    ]
    return JsonResponse({"messages": messages})


@csrf_exempt
@require_http_methods(["GET"])
def commands_view(request):
    """List device commands."""
    status = request.GET.get("status", "all")
    qs = DeviceCommand.objects.all()
    if status != "all":
        qs = qs.filter(status=status)
    commands = [
        {
            "id": str(c.id),
            "device_type": c.device_type,
            "device_name": c.device_name,
            "command": c.command,
            "parameters": c.parameters,
            "status": c.status,
            "requires_confirmation": c.requires_confirmation,
            "confirmed": c.confirmed,
            "created_at": c.created_at.isoformat(),
            "executed_at": c.executed_at.isoformat() if c.executed_at else None,
        }
        for c in qs[:50]
    ]
    return JsonResponse({"commands": commands})


@csrf_exempt
@require_http_methods(["POST"])
def confirm_command_view(request, command_id):
    """Confirm a pending device command."""
    try:
        cmd = DeviceCommand.objects.get(id=command_id)
    except DeviceCommand.DoesNotExist:
        return JsonResponse({"error": "Command not found"}, status=404)

    if cmd.status != "pending":
        return JsonResponse({"error": f"Command is {cmd.status}, not pending"}, status=400)

    cmd.confirmed = True
    cmd.status = "confirmed"
    cmd.save()

    # In a real system, this would trigger actual device control
    # For now, mark as executed
    from django.utils import timezone
    cmd.status = "executed"
    cmd.executed_at = timezone.now()
    cmd.result = "Simulated execution successful"
    cmd.save()

    return JsonResponse({"id": str(cmd.id), "status": cmd.status, "result": cmd.result})


@csrf_exempt
@require_http_methods(["GET"])
def action_log_view(request):
    """View audit trail of all actions."""
    action_type = request.GET.get("type", "all")
    qs = ActionLog.objects.all()
    if action_type != "all":
        qs = qs.filter(action_type=action_type)
    logs = [
        {
            "id": str(l.id),
            "action_type": l.action_type,
            "action_id": str(l.action_id),
            "transcript": l.transcript,
            "result": l.result,
            "created_at": l.created_at.isoformat(),
        }
        for l in qs[:100]
    ]
    return JsonResponse({"logs": logs})
