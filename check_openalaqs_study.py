import argparse
import sqlite3
from pathlib import Path


TABLES = [
    "shapes_buildings",
    "shapes_runways",
    "shapes_taxiways",
    "shapes_gates",
    "shapes_roadways",
    "shapes_parking",
    "shapes_area_sources",
    "shapes_point_sources",
    "user_study_setup",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a .alaqs study before running the DeepSDF pipeline.")
    parser.add_argument("study", nargs="?", default="data_sources/airport_study.alaqs")
    args = parser.parse_args()

    study_path = Path(args.study)
    if not study_path.exists():
        raise FileNotFoundError(f"Study not found: {study_path}")

    print(f"Study: {study_path.resolve()}")
    print(f"Size: {study_path.stat().st_size / (1024 * 1024):.2f} MB")

    with sqlite3.connect(study_path) as conn:
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in cur.fetchall()}
        print(f"Tables detected: {len(tables)}")

        counts = {}
        for table in TABLES:
            if table not in tables:
                counts[table] = None
                continue
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = int(cur.fetchone()[0])

        print("\nMain layers:")
        for table in TABLES:
            print(f"  {table}: {counts[table]}")

        status_ok = True
        if counts["shapes_buildings"] in (None, 0):
            status_ok = False
            print("\nWarning: buildings layer is empty or missing.")
        if counts["shapes_runways"] in (None, 0):
            status_ok = False
            print("Warning: runways layer is empty or missing.")
        if counts["shapes_taxiways"] in (None, 0):
            status_ok = False
            print("Warning: taxiways layer is empty or missing.")

        print("\nStatus:", "OK" if status_ok else "INCOMPLETE")


if __name__ == "__main__":
    main()
