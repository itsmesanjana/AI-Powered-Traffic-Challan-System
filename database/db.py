from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "traffic_violation_system.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_database (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_name TEXT NOT NULL,
            mobile_number TEXT NOT NULL,
            four_wheeler_number TEXT,
            two_wheeler_number TEXT,
            location TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS violation_database (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_type TEXT NOT NULL,
            violation_name TEXT NOT NULL,
            amount INTEGER NOT NULL,
            UNIQUE(vehicle_type, violation_name)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS challan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            owner_name TEXT,
            mobile_number TEXT,
            vehicle_type TEXT,
            violation TEXT NOT NULL,
            amount INTEGER NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def fetch_one(query: str, params: tuple = ()) -> Dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute(query, params).fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_all(query: str, params: tuple = ()) -> list[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def execute_query(query: str, params: tuple = ()) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id
