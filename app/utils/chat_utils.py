from datetime import datetime
from typing import Any, Optional
from bson import ObjectId
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "format_timestamp",
    "parse_timestamp",
    "format_message_for_socket",
    "format_messages_for_display",
    "validate_message",
]

def format_timestamp(timestamp: datetime) -> str:
    try:
        return timestamp.strftime('%H:%M')
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return "00:00"

def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.error(f"Failed to parse timestamp: {timestamp_str}")
            return datetime.now()

def format_message_for_socket(message: dict[str, Any]) -> dict[str, Any]:
    try:
        message_copy = message.copy()
        if isinstance(message_copy.get('timestamp'), datetime):
            message_copy['timestamp'] = message_copy['timestamp'].isoformat()
        if isinstance(message_copy.get('_id'), ObjectId):
            message_copy['_id'] = str(message_copy['_id'])
        return message_copy
    except Exception as e:
        logger.error(f"Error formatting message for socket: {e}")
        return message

def format_messages_for_display(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for msg in messages:
        sender_name = msg.get("sender") or msg.get("username", "Unknown")
        message_text = msg.get("content") or msg.get("message", "")
        timestamp = msg.get("timestamp", datetime.now())

        if isinstance(timestamp, str):
            timestamp = parse_timestamp(timestamp)

        formatted.append({
            "sender": sender_name,
            "text": message_text,
            "timestamp": format_timestamp(timestamp),
        })
    return formatted

def validate_message(message: dict[str, Any]) -> Optional[str]:
    try:
        message_text = message.get('content') or message.get('message')
        
        if not isinstance(message_text, str):
            return "Message must be a string"
        if not message_text.strip():
            return "Message cannot be empty"
        if not isinstance(message.get('username'), str):
            return "Username must be a string"
        if not isinstance(message.get('room'), str):
            return "Room must be a string"
        return None
    except Exception as e:
        logger.error(f"Error validating message: {e}")
        return "Invalid message format"


