from __future__ import annotations

from typing import Any, Dict

from database.db import fetch_all, fetch_one
from utils.text_utils import normalize_plate, normalize_violation_name


def get_owner_by_plate(plate_number: str) -> Dict[str, Any] | None:
    normalized = normalize_plate(plate_number)
    if not normalized:
        return None

    rows = fetch_all("SELECT * FROM vehicle_database")
    for row in rows:
        four = normalize_plate(row.get("four_wheeler_number"))
        two = normalize_plate(row.get("two_wheeler_number"))
        if normalized and (normalized == four or normalized == two):
            return row
    return None


def get_violation_amount(vehicle_type: str, violation_name: str) -> int | None:
    vehicle_type = (vehicle_type or "").lower().strip()
    violation_name = normalize_violation_name(violation_name)
    row = fetch_one(
        "SELECT amount FROM violation_database WHERE vehicle_type = ? AND violation_name = ?",
        (vehicle_type, violation_name),
    )
    return int(row["amount"]) if row else None


def match_violation_in_db(vehicle_type: str, violation_name: str) -> Dict[str, Any] | None:
    vehicle_type = (vehicle_type or "").lower().strip()
    violation_name = normalize_violation_name(violation_name)
    return fetch_one(
        "SELECT * FROM violation_database WHERE vehicle_type = ? AND violation_name = ?",
        (vehicle_type, violation_name),
    )


def fetch_recent_logs(limit: int = 10) -> list[Dict[str, Any]]:
    return fetch_all("SELECT * FROM challan_logs ORDER BY id DESC LIMIT ?", (limit,))
