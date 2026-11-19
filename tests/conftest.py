from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHINOOK_PATH = PROJECT_ROOT / "data" / "Chinook_Sqlite.sqlite"
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge"
CASES_PATH = PROJECT_ROOT / "evaluation" / "cases.json"


@pytest.fixture
def sample_database(tmp_path: Path) -> Path:
    """A tiny database with two related tables, independent of the Chinook file."""
    path = tmp_path / "sample.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT NOT NULL);
            CREATE TABLE Album (
                AlbumId INTEGER PRIMARY KEY,
                Title TEXT NOT NULL,
                ArtistId INTEGER NOT NULL REFERENCES Artist (ArtistId)
            );
            INSERT INTO Artist (ArtistId, Name) VALUES
                (1, 'Example Artist'), (2, 'Second Artist'), (3, 'Third Artist');
            INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (10, 'First Album', 1);
            """
        )
    connection.close()
    return path


@pytest.fixture
def chinook_database(tmp_path: Path) -> Path:
    """A private copy of the real Chinook database (tests never touch the shipped file)."""
    if not CHINOOK_PATH.is_file():
        pytest.skip(f"Chinook database not found at {CHINOOK_PATH}")
    copy = tmp_path / "chinook.sqlite"
    shutil.copyfile(CHINOOK_PATH, copy)
    return copy
