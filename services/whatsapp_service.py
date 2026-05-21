from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv

from database.db import execute_query

load_dotenv()


def generate_whatsapp_message(owner: str, plate: str, violation: str, amount: int, location: str) -> str:
    return (
        "Traffic Violation Alert!\n\n"
        f"Hello {owner},\n"
        f"Your vehicle ({plate}) was caught for {violation} in {location}.\n"
        f"Please pay the fine of ₹{amount}.\n\n"
        "Drive safe!"
    )


def send_whatsapp_message(target_number: str, message: str) -> Dict[str, Any]:
    mode = os.getenv("WHATSAPP_MODE", "demo").lower().strip()

    if mode == "twilio":
        try:
            from twilio.rest import Client
        except Exception as exc:
            return {"status": "failed", "error": f"Twilio not installed: {exc}"}

        account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        from_number = os.getenv("TWILIO_WHATSAPP_FROM", "")
        to_number = os.getenv("TWILIO_WHATSAPP_TO", target_number)
        if not account_sid or not auth_token or not from_number:
            return {"status": "failed", "error": "Missing Twilio WhatsApp credentials in .env"}

        client = Client(account_sid, auth_token)
        try:
            sent = client.messages.create(
                body=message,
                from_=from_number,
                to=to_number,
            )
            return {"status": "sent", "provider": "twilio", "sid": sent.sid, "target_number": to_number}
        except Exception as exc:
            return {"status": "failed", "provider": "twilio", "error": str(exc), "target_number": to_number}

    return {"status": "sent_demo", "provider": "demo", "target_number": target_number}


def save_challan_log(
    plate_number: str,
    owner_name: str,
    mobile_number: str,
    vehicle_type: str,
    violation: str,
    amount: int,
    message: str,
    status: str,
) -> int:
    return execute_query(
        """
        INSERT INTO challan_logs
        (plate_number, owner_name, mobile_number, vehicle_type, violation, amount, message, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (plate_number, owner_name, mobile_number, vehicle_type, violation, amount, message, status),
    )
