"""
Action handlers for Command Center Pipeline v2.

Routes parsed intents of type action_* to the appropriate handler,
persists the action in the database, and returns a voice confirmation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from django.utils import timezone

from actions.models import Reminder, Message, DeviceCommand, ActionLog
from layer2.intent_parser import ParsedIntent

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    voice_response: str
    action_type: str = ""
    action_id: Optional[str] = None


class ActionHandler:
    """Executes actions based on parsed intent and persists to DB."""

    def execute(self, intent: ParsedIntent) -> ActionResult:
        """Route to the appropriate handler based on intent type."""
        handlers = {
            "action_reminder": self._create_reminder,
            "action_message": self._send_message,
            "action_control": self._device_command,
            "action_task": self._create_task,
        }

        handler = handlers.get(intent.type)
        if not handler:
            return ActionResult(
                success=False,
                voice_response="I'm not sure how to handle that action. Could you rephrase?",
            )

        try:
            result = handler(intent)
            # Log the action
            ActionLog.objects.create(
                action_type=intent.type.replace("action_", ""),
                action_id=result.action_id or "00000000-0000-0000-0000-000000000000",
                transcript=intent.raw_text,
                intent_json={
                    "type": intent.type,
                    "domains": intent.domains,
                    "entities": intent.entities,
                    "parameters": intent.parameters,
                },
                result="success" if result.success else "failed",
            )
            return result
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return ActionResult(
                success=False,
                voice_response="Sorry, I couldn't complete that action. Please try again.",
            )

    def _create_reminder(self, intent: ParsedIntent) -> ActionResult:
        """Create a scheduled reminder."""
        params = intent.parameters
        message = params.get("message", intent.raw_text)
        entity = ""
        if intent.entities.get("devices"):
            entity = intent.entities["devices"][0]

        # Parse trigger time
        trigger_time = self._parse_trigger_time(params)

        reminder = Reminder.objects.create(
            message=message,
            trigger_time=trigger_time,
            recurring=params.get("recurring", ""),
            entity=entity,
        )

        time_str = trigger_time.strftime("%I:%M %p")
        return ActionResult(
            success=True,
            voice_response=f"Reminder set for {time_str}: {message[:60]}",
            action_type="reminder",
            action_id=str(reminder.id),
        )

    def _send_message(self, intent: ParsedIntent) -> ActionResult:
        """Queue a message to a recipient."""
        params = intent.parameters
        recipient = params.get("recipient", "operations team")
        content = params.get("content", intent.raw_text)
        channel = params.get("channel", "internal")

        msg = Message.objects.create(
            recipient=recipient,
            content=content,
            channel=channel,
        )

        return ActionResult(
            success=True,
            voice_response=f"Message sent to {recipient}: {content[:60]}",
            action_type="message",
            action_id=str(msg.id),
        )

    def _device_command(self, intent: ParsedIntent) -> ActionResult:
        """Issue a device control command (requires confirmation for safety)."""
        params = intent.parameters
        entities = intent.entities

        device_name = ""
        device_type = ""
        if entities.get("devices"):
            device_ref = entities["devices"][0]
            parts = device_ref.rsplit("_", 1)
            device_type = parts[0] if parts else device_ref
            device_name = device_ref

        command = params.get("command", "unknown")
        # Infer command from transcript if not in params
        if command == "unknown":
            text = intent.raw_text.lower()
            if "start" in text or "turn on" in text:
                command = "start"
            elif "stop" in text or "turn off" in text:
                command = "stop"
            elif "set" in text or "adjust" in text:
                command = "set_parameter"
            else:
                command = "unknown"

        cmd = DeviceCommand.objects.create(
            device_type=device_type,
            device_name=device_name,
            command=command,
            parameters=params,
            requires_confirmation=True,
        )

        return ActionResult(
            success=True,
            voice_response=(
                f"Command '{command}' for {device_name or 'device'} is pending confirmation. "
                f"Please confirm to proceed."
            ),
            action_type="command",
            action_id=str(cmd.id),
        )

    def _create_task(self, intent: ParsedIntent) -> ActionResult:
        """Create a task / work order (stored as DeviceCommand with type 'task')."""
        params = intent.parameters
        entities = intent.entities

        device_name = ""
        if entities.get("devices"):
            device_name = entities["devices"][0]

        description = params.get("description", intent.raw_text)

        cmd = DeviceCommand.objects.create(
            device_type="task",
            device_name=device_name,
            command="create_work_order",
            parameters={"description": description, **params},
            requires_confirmation=False,
            status="executed",
        )

        return ActionResult(
            success=True,
            voice_response=f"Work order created: {description[:60]}",
            action_type="task",
            action_id=str(cmd.id),
        )

    def _parse_trigger_time(self, params: dict) -> datetime:
        """Parse trigger time from parameters. Falls back to 1 hour from now."""
        time_str = params.get("time", "")
        now = timezone.now()

        if not time_str:
            return now + timedelta(hours=1)

        # Try common formats
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
            try:
                parsed = datetime.strptime(time_str.strip(), fmt)
                trigger = now.replace(
                    hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0,
                )
                # If the time is in the past today, schedule for tomorrow
                if trigger <= now:
                    trigger += timedelta(days=1)
                return trigger
            except ValueError:
                continue

        # Relative time: "in 30 minutes", "in 2 hours"
        import re
        match = re.search(r'in\s+(\d+)\s*(minute|hour|min|hr)', time_str.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("hour") or unit.startswith("hr"):
                return now + timedelta(hours=amount)
            else:
                return now + timedelta(minutes=amount)

        return now + timedelta(hours=1)
