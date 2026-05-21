from __future__ import annotations

from database.db import execute_query, fetch_one

VEHICLE_SAMPLE_DATA = [
    ("Sanjana Kanaki", "9067443576", "MH13 AB1234", None, "Maharashtra"),
    ("Rahul Singh", "9067443576", None, "6550VB", "Maharashtra"),
]

VIOLATION_SAMPLE_DATA = [
    ("bike", "no helmet", 500),
    ("bike", "overspeed", 1000),
    ("bike", "triple riding", 1500),
    ("truck", "overspeed", 1000),
    ("car", "overspeed", 1000),
]


def seed_vehicle_data(force_reset: bool = False) -> None:
    if force_reset:
        execute_query("DELETE FROM vehicle_database")

    existing = fetch_one("SELECT id FROM vehicle_database LIMIT 1")
    if existing:
        return

    for row in VEHICLE_SAMPLE_DATA:
        execute_query(
            """
            INSERT INTO vehicle_database
            (owner_name, mobile_number, four_wheeler_number, two_wheeler_number, location)
            VALUES (?, ?, ?, ?, ?)
            """,
            row,
        )


def seed_violation_data(force_reset: bool = False) -> None:
    if force_reset:
        execute_query("DELETE FROM violation_database")

    existing = fetch_one("SELECT id FROM violation_database LIMIT 1")
    if existing:
        return

    for row in VIOLATION_SAMPLE_DATA:
        execute_query(
            """
            INSERT INTO violation_database
            (vehicle_type, violation_name, amount)
            VALUES (?, ?, ?)
            """,
            row,
        )
