from __future__ import annotations

import json
import re
from typing import Any, Dict


def normalize_plate(plate: str | None) -> str:
    if not plate:
        return ""
    plate = str(plate).upper()
    plate = plate.replace(" ", "")
    plate = plate.replace("\n", "")
    plate = plate.replace("\t", "")
    plate = re.sub(r"[^A-Z0-9]", "", plate)
    return plate


def normalize_violation_name(violation: str | None) -> str:
    if not violation:
        return "unknown"
    violation = str(violation).strip().lower().replace("_", " ")
    return " ".join(violation.split())


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}
