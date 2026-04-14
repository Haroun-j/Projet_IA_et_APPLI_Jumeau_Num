"""
Lecture des études OpenALAQS
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from shapely import wkb
from shapely.geometry import mapping
from shapely.ops import transform


LAYER_SPECS: dict[str, dict[str, Any]] = {
    "buildings": {
        "table": "shapes_buildings",
        "id_col": "building_id",
        "height_col": "height",
    },
    "gates": {
        "table": "shapes_gates",
        "id_col": "gate_id",
        "height_col": "gate_height",
    },
    "runways": {
        "table": "shapes_runways",
        "id_col": "runway_id",
        "height_col": None,
    },
    "taxiways": {
        "table": "shapes_taxiways",
        "id_col": "taxiway_id",
        "height_col": None,
    },
    "roadways": {
        "table": "shapes_roadways",
        "id_col": "roadway_id",
        "height_col": "height",
    },
    "parking": {
        "table": "shapes_parking",
        "id_col": "parking_id",
        "height_col": "height",
    },
    "area_sources": {
        "table": "shapes_area_sources",
        "id_col": "source_id",
        "height_col": "height",
    },
    "point_sources": {
        "table": "shapes_point_sources",
        "id_col": "source_id",
        "height_col": "height",
    },
}


def latlon_to_webmercator(lat: float, lon: float) -> tuple[float, float]:
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        [table_name],
    ).fetchone()
    return row is not None


def _truthy_instudy(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no", "n"}


def _spatialite_blob_to_wkb(blob: Any) -> bytes:
    if blob is None:
        raise ValueError("Empty geometry blob.")
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(f"Unsupported geometry type: {type(blob)!r}")

    data = bytes(blob)
    if not data:
        raise ValueError("Empty geometry blob.")

    # OpenALAQS stores geometries as SpatiaLite blobs:
    # 00 | endian | srid | bbox(4 doubles) | 7C | WKB | FE
    if data[0] == 0x00:
        marker_idx = 38 if len(data) > 39 and data[38] == 0x7C else -1
        if marker_idx < 0:
            for idx in range(32, min(len(data), 48)):
                if data[idx] == 0x7C:
                    marker_idx = idx
                    break
        if marker_idx < 0:
            raise ValueError("Could not locate SpatiaLite geometry marker.")
        wkb_bytes = data[marker_idx + 1 :]
        if wkb_bytes and wkb_bytes[-1] == 0xFE:
            wkb_bytes = wkb_bytes[:-1]
        return wkb_bytes

    return data


def _load_spatialite_geometry(blob: Any):
    return wkb.loads(_spatialite_blob_to_wkb(blob))


def _shift_to_local(geom, origin_x: float, origin_y: float):
    return transform(
        lambda x, y, z=None: (
            (x - origin_x, y - origin_y)
            if z is None
            else (x - origin_x, y - origin_y, z)
        ),
        geom,
    )


def _serialize_feature(layer_name: str, feature_id: str, geom, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "layer": layer_name,
            "id": feature_id,
            **props,
        },
    }


def _read_layer(
    conn: sqlite3.Connection,
    layer_name: str,
    *,
    table_name: str,
    id_col: str,
    height_col: str | None,
    origin_x: float,
    origin_y: float,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, table_name):
        return []

    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    features: list[dict[str, Any]] = []

    for row in rows:
        row_dict = dict(row)
        if "instudy" in row_dict and not _truthy_instudy(row_dict["instudy"]):
            continue
        geom_blob = row_dict.get("geometry")
        if geom_blob is None:
            continue

        try:
            geom = _load_spatialite_geometry(geom_blob)
        except Exception:
            continue
        if geom.is_empty:
            continue

        local_geom = _shift_to_local(geom, origin_x, origin_y)
        feature_id = row_dict.get(id_col) or f"{table_name}_{row_dict.get('oid', len(features))}"
        props = {"source_table": table_name}
        if height_col and row_dict.get(height_col) is not None:
            try:
                props["height"] = float(row_dict[height_col])
            except (TypeError, ValueError):
                pass

        features.append(
            {
                "id": str(feature_id),
                "geometry": local_geom,
                "properties": props,
                "raw": row_dict,
            }
        )

    return features


def load_openalaqs_study(
    db_path: str,
    *,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> dict[str, Any]:
    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"OpenALAQS database not found: {db_file}")

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        study_setup = conn.execute("SELECT * FROM user_study_setup LIMIT 1").fetchone()
        study_info = dict(study_setup) if study_setup is not None else {}

        lat = center_lat
        lon = center_lon
        if lat is None and study_info.get("airport_latitude") is not None:
            lat = float(study_info["airport_latitude"])
        if lon is None and study_info.get("airport_longitude") is not None:
            lon = float(study_info["airport_longitude"])
        if lat is None or lon is None:
            raise ValueError(
                "Could not determine study origin. Provide center_lat and center_lon in the config."
            )

        origin_x, origin_y = latlon_to_webmercator(lat, lon)

        layers: dict[str, list[dict[str, Any]]] = {}
        for layer_name, spec in LAYER_SPECS.items():
            layers[layer_name] = _read_layer(
                conn,
                layer_name,
                table_name=spec["table"],
                id_col=spec["id_col"],
                height_col=spec["height_col"],
                origin_x=origin_x,
                origin_y=origin_y,
            )

        return {
            "db_path": str(db_file),
            "study_info": study_info,
            "origin": {
                "center_lat": lat,
                "center_lon": lon,
                "origin_x_3857": origin_x,
                "origin_y_3857": origin_y,
            },
            "layers": layers,
            "layer_counts": {name: len(features) for name, features in layers.items()},
        }
    finally:
        conn.close()


def export_openalaqs_reference(study_data: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_collection = {"type": "FeatureCollection", "features": []}
    summary = {
        "db_path": study_data["db_path"],
        "study_info": study_data["study_info"],
        "origin": study_data["origin"],
        "layer_counts": study_data["layer_counts"],
        "layers": {},
    }

    for layer_name, features in study_data["layers"].items():
        summary["layers"][layer_name] = []
        for feat in features:
            record = _serialize_feature(
                layer_name,
                feat["id"],
                feat["geometry"],
                feat["properties"],
            )
            feature_collection["features"].append(record)
            summary["layers"][layer_name].append(
                {
                    "id": feat["id"],
                    "geometry_type": feat["geometry"].geom_type,
                    "bounds": [float(v) for v in feat["geometry"].bounds],
                    **feat["properties"],
                }
            )

    geojson_path = out_dir / "openalaqs_reference_layers.geojson"
    summary_path = out_dir / "openalaqs_reference_layers.json"

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "geojson": str(geojson_path),
        "summary": str(summary_path),
    }
