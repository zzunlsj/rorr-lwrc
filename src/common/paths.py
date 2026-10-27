"""프로젝트 경로 상수 (CLAUDE.md §4)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
L1 = DATA / "l1"
L2 = DATA / "l2"
L3 = DATA / "l3"
TMP = DATA / "tmp"

MANIFEST = ROOT / "manifest"
SPECS = ROOT / "specs"
OUTPUTS = ROOT / "outputs"
REFERENCE = ROOT / "reference"

TOURNAMENTS = (
    "LCK_Cup",
    "FirstStand",
    "LCK_R1-2",
    "Road_to_MSI",
    "MSI",
    "LCK_R3-4",
    "LCK_PlayIn_PO",
    "Worlds",
)
