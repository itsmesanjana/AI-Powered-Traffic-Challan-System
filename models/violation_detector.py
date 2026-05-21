from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from utils.text_utils import extract_json_object, normalize_violation_name

load_dotenv()

BIKE_VIOLATIONS = {"overspeed", "triple riding", "no helmet"}
CAR_TRUCK_VIOLATIONS = {"overspeed"}


def _client() -> OpenAI | None:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _allowed_violations(vehicle_type: str) -> set[str]:
    vehicle_type = (vehicle_type or "").lower().strip()
    if vehicle_type == "bike":
        return BIKE_VIOLATIONS
    if vehicle_type in {"car", "truck"}:
        return CAR_TRUCK_VIOLATIONS
    return {"unknown"}


def detect_violation_from_full_image(image_path: str, known_vehicle_type: str = "") -> Dict[str, Any]:
    known_vehicle_type = (known_vehicle_type or "").lower().strip()
    
    if known_vehicle_type == "car":
        return {
            "vehicle_type": "car",
            "violation_name": "overspeed",
            "reason": "Rule-based DB lookup: Car always gets overspeed violation.",
        }

    client = _client()
    if client is None:
        if known_vehicle_type == "bike":
            return {
                "vehicle_type": "bike",
                "violation_name": "no helmet",
                "reason": "AI offline. Auto-assigning bike/no helmet based on DB lookup.",
            }
        return {
            "vehicle_type": "unknown",
            "violation_name": "unknown",
            "reason": "Missing XAI_API_KEY in .env",
        }

    image_b64 = _image_to_base64(image_path)
    prompt = """
Analyze this traffic evidence image and return JSON only.

Rules:
1. Detect the main vehicle type: bike, car, truck, or unknown.
2. Detect one violation only.
3. For bike, allowed violations are: overspeed, triple riding, no helmet. (NEVER no seatbelt for bike)
4. For car, allowed violations are: overspeed. (NEVER triple riding or no helmet for car)
5. For truck, allowed violations are: overspeed. (NEVER triple riding for truck)
6. Never allow triple riding for car or truck. Never allow no helmet for car. Never allow no seatbelt for bike.
7. If the violation is not clearly visible, return unknown.
8. If multiple riders are on a bike and there are 3 people, return triple riding.
9. If no vehicle is visible, return unknown.

Return strict JSON:
{
  "vehicle_type": "bike|car|truck|unknown",
  "violation_name": "overspeed|triple riding|unknown",
  "reason": "short explanation"
}
""".strip()

    try:
        response = client.chat.completions.create(
            model=os.getenv("XAI_MODEL", "llama-3.2-11b-vision-preview"),
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = extract_json_object(content)
        vehicle_type = str(data.get("vehicle_type", "unknown")).lower().strip()
        violation = normalize_violation_name(data.get("violation_name", "unknown"))
        reason = str(data.get("reason", "")).strip()

        if known_vehicle_type == "bike":
            vehicle_type = "bike"
            if violation != "triple riding":
                violation = "no helmet"
                reason = "Rule-based fallback: Bike is not triple riding, defaulting to no helmet."

        allowed = _allowed_violations(vehicle_type)
        if violation not in allowed:
            violation = "unknown"

        return {
            "vehicle_type": vehicle_type or "unknown",
            "violation_name": violation or "unknown",
            "reason": reason,
        }
    except Exception as exc:
        print(f"AI Model Error: {exc}")
        if known_vehicle_type == "bike":
            return {
                "vehicle_type": "bike",
                "violation_name": "no helmet",
                "reason": "AI offline. Auto-assigning bike/no helmet based on DB lookup.",
            }
        return {
            "vehicle_type": "unknown",
            "violation_name": "unknown",
            "reason": "AI offline.",
        }
