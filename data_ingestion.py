"""
 data ingestion and 3D extrusion for the airport DeepSDF pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import requests
import trimesh
import yaml
from shapely.geometry import Polygon, shape
from shapely.ops import orient as shapely_orient

from openalaqs_import import export_openalaqs_reference, load_openalaqs_study


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def latlon_to_local_xy(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    """Project geographic coordinates onto a local East-North plane in metres."""
    radius = 6_371_000.0
    x = radius * math.radians(lon - center_lon) * math.cos(math.radians(center_lat))
    y = radius * math.radians(lat - center_lat)
    return x, y


def latlon_to_webmercator(lat: float, lon: float) -> tuple[float, float]:
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


def webmercator_to_local_xy(x: float, y: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    cx, cy = latlon_to_webmercator(center_lat, center_lon)
    return x - cx, y - cy


def overpass_query(lat: float, lon: float, radius_m: float) -> dict:
    """
    Fetch terminal/building footprints from Overpass.

    The query keeps generic airport buildings, but also includes
    aeroway=terminal to stay closer to the OpenALAQS workflow.
    """
    query = f"""
[out:json][timeout:120];
(
  way["building"](around:{radius_m},{lat},{lon});
  relation["building"](around:{radius_m},{lat},{lon});
  way["aeroway"="terminal"](around:{radius_m},{lat},{lon});
  relation["aeroway"="terminal"](around:{radius_m},{lat},{lon});
);
out body;
>;
out skel qt;
"""
    url = "https://overpass-api.de/api/interpreter"
    print(f"  Querying Overpass API (radius={radius_m} m)...")
    for attempt in range(3):
        try:
            response = requests.post(url, data={"data": query}, timeout=130)
            response.raise_for_status()
            data = response.json()
            print(f"  Got {len(data.get('elements', []))} OSM elements.")
            return data
        except Exception as exc:
            print(f"  Attempt {attempt + 1}/3 failed: {exc}")
            time.sleep(5)
    raise RuntimeError("Overpass API unreachable after 3 attempts.")


def _infer_height_from_properties(props: dict, default_height: float, height_per_level: float) -> float:
    try:
        if "height" in props and props["height"] not in (None, ""):
            return max(float(str(props["height"]).replace("m", "").strip()), 3.0)
        for key in ("building:levels", "levels"):
            if key in props and props[key] not in (None, ""):
                return max(float(props[key]) * height_per_level, 3.0)
    except (ValueError, TypeError):
        pass
    return max(default_height, 3.0)


def parse_geojson_buildings(
    geojson_path: str,
    center_lat: float,
    center_lon: float,
    default_height: float,
    height_per_level: float,
    source_epsg: int = 4326,
) -> list[tuple[Polygon, float]]:
    """
    Load polygons from a local GeoJSON / QGIS export.

    Supported coordinate systems:
    - EPSG:4326 -> lon/lat projected to local XY
    - EPSG:3857 -> shifted to local XY around airport centre
    - other -> coordinates are used as-is
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    buildings: list[tuple[Polygon, float]] = []
    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue
        props = feature.get("properties", {}) or {}

        try:
            shp = shape(geom)
        except Exception:
            continue

        if shp.geom_type == "Polygon":
            polygons = [shp]
        elif shp.geom_type == "MultiPolygon":
            polygons = list(shp.geoms)
        else:
            polygons = []

        height = _infer_height_from_properties(props, default_height, height_per_level)

        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                continue
            coords = list(polygon.exterior.coords)
            if len(coords) < 4:
                continue

            xy_coords: list[tuple[float, float]] = []
            for x, y in coords:
                if source_epsg == 4326 or (abs(x) <= 180 and abs(y) <= 90):
                    lx, ly = latlon_to_local_xy(y, x, center_lat, center_lon)
                elif source_epsg == 3857:
                    lx, ly = webmercator_to_local_xy(float(x), float(y), center_lat, center_lon)
                else:
                    lx, ly = float(x), float(y)
                xy_coords.append((lx, ly))

            try:
                local_polygon = Polygon(xy_coords)
            except Exception:
                continue

            if local_polygon.is_valid and not local_polygon.is_empty and local_polygon.area > 1.0:
                local_polygon = shapely_orient(local_polygon, sign=1.0)
                buildings.append((local_polygon, height))

    print(f"  Parsed {len(buildings)} valid buildings from GeoJSON.")
    return buildings


def parse_osm_buildings(
    data: dict,
    center_lat: float,
    center_lon: float,
    default_height: float,
    height_per_level: float,
) -> list[tuple[Polygon, float]]:
    """Return a list of terminal/building polygons from OSM JSON."""
    nodes: dict[int, tuple[float, float]] = {}
    for element in data["elements"]:
        if element["type"] == "node":
            x, y = latlon_to_local_xy(element["lat"], element["lon"], center_lat, center_lon)
            nodes[element["id"]] = (x, y)

    buildings: list[tuple[Polygon, float]] = []
    for element in data["elements"]:
        if element["type"] != "way":
            continue

        tags = element.get("tags", {})
        is_building = "building" in tags
        is_terminal = tags.get("aeroway") == "terminal"
        if not (is_building or is_terminal):
            continue

        coords = [nodes[node_id] for node_id in element.get("nodes", []) if node_id in nodes]
        if len(coords) < 3:
            continue

        try:
            if "height" in tags:
                height = float(tags["height"].replace("m", "").strip())
            elif "building:levels" in tags:
                height = float(tags["building:levels"]) * height_per_level
            elif is_terminal:
                height = max(default_height, 12.0)
            else:
                height = default_height
        except (ValueError, TypeError):
            height = default_height

        height = max(height, 3.0)

        try:
            polygon = Polygon(coords)
        except Exception:
            continue

        if polygon.is_valid and not polygon.is_empty and polygon.area > 1.0:
            polygon = shapely_orient(polygon, sign=1.0)
            buildings.append((polygon, height))

    print(f"  Parsed {len(buildings)} valid OSM terminal/building polygons.")
    return buildings


def extrude_polygon_to_mesh(polygon: Polygon, height: float):
    """Extrude a 2D polygon into a watertight trimesh."""
    coords = list(polygon.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    n_vertices = len(coords)
    if n_vertices < 3:
        return None

    verts = np.array(
        [[x, y, 0.0] for x, y in coords] + [[x, y, height] for x, y in coords],
        dtype=np.float32,
    )

    faces = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        faces += [[i, j, n_vertices + j], [i, n_vertices + j, n_vertices + i]]

    for i in range(1, n_vertices - 1):
        faces.append([0, i + 1, i])

    for i in range(1, n_vertices - 1):
        faces.append([n_vertices, n_vertices + i, n_vertices + i + 1])

    return trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int32),
        process=True,
    )


def create_test_cube(mesh_dir: str) -> None:
    """Create a simple 10 x 20 x 15 m box at the origin for quick tests."""
    verts = np.array(
        [
            [-5, -10, 0],
            [5, -10, 0],
            [5, 10, 0],
            [-5, 10, 0],
            [-5, -10, 15],
            [5, -10, 15],
            [5, 10, 15],
            [-5, 10, 15],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    output_path = os.path.join(mesh_dir, "test_cube.obj")
    mesh.export(output_path)
    print(f"  Test cube saved -> {output_path}")


def _resolve_optional_path(base_dir: str, value: str) -> str:
    if not value:
        return ""
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def _buildings_from_openalaqs(
    db_path: str,
    airport_cfg: dict,
    mesh_dir: str,
) -> tuple[list[tuple[Polygon, float]], dict | None]:
    alaqs_data = load_openalaqs_study(
        db_path,
        center_lat=airport_cfg.get("center_lat"),
        center_lon=airport_cfg.get("center_lon"),
    )
    reference_files = export_openalaqs_reference(alaqs_data, mesh_dir)
    print(f"  OpenALAQS layer counts: {alaqs_data['layer_counts']}")

    buildings: list[tuple[Polygon, float]] = []
    for feature in alaqs_data["layers"]["buildings"]:
        geom = feature["geometry"]
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            polygons = []

        for polygon in polygons:
            if polygon.is_valid and not polygon.is_empty and polygon.area > 1.0:
                polygon = shapely_orient(polygon, sign=1.0)
                height = float(feature["properties"].get("height", airport_cfg["default_height_m"]))
                buildings.append((polygon, max(height, 3.0)))

    return buildings, {
        "study": alaqs_data,
        "files": reference_files,
    }


def run_ingestion(config_path: str, use_test_cube: bool = False) -> None:
    cfg = load_config(config_path)
    airport = cfg["airport"]
    paths = cfg["paths"]

    base_dir = os.path.dirname(os.path.abspath(config_path))
    mesh_dir = os.path.join(base_dir, paths["meshes"])
    os.makedirs(mesh_dir, exist_ok=True)

    if use_test_cube:
        print("=== TEST MODE: generating synthetic cube ===")
        create_test_cube(mesh_dir)
        return

    source_type = "osm"
    buildings: list[tuple[Polygon, float]] = []
    openalaqs_context = None

    openalaqs_db = _resolve_optional_path(
        base_dir,
        airport.get("open_alaqs_database") or airport.get("open_alaqs_study", ""),
    )
    geojson_path = _resolve_optional_path(base_dir, airport.get("footprints_geojson", ""))
    footprints_epsg = int(airport.get("footprints_crs_epsg", 4326))

    if openalaqs_db:
        print(f"=== Loading OpenALAQS study for {airport['name']} ===")
        print(f"  Database source: {openalaqs_db}")
        if not os.path.isfile(openalaqs_db):
            raise FileNotFoundError(f"OpenALAQS database not found: {openalaqs_db}")

        buildings, openalaqs_context = _buildings_from_openalaqs(openalaqs_db, airport, mesh_dir)
        if buildings:
            source_type = "open_alaqs_database"
            print(f"  Using {len(buildings)} building polygons from OpenALAQS.")
        else:
            print("  No buildings found in OpenALAQS. Falling back to the next available source.")

    if not buildings and geojson_path:
        print(f"=== Loading local building footprints for {airport['name']} ===")
        print(f"  GeoJSON source: {geojson_path}")
        if not os.path.isfile(geojson_path):
            raise FileNotFoundError(f"GeoJSON source not found: {geojson_path}")
        buildings = parse_geojson_buildings(
            geojson_path,
            airport["center_lat"],
            airport["center_lon"],
            airport["default_height_m"],
            airport["height_per_level_m"],
            source_epsg=footprints_epsg,
        )
        source_type = "geojson"

    if not buildings:
        print(f"=== Downloading terminal/building footprints for {airport['name']} ===")
        osm_data = overpass_query(airport["center_lat"], airport["center_lon"], airport["radius_m"])
        buildings = parse_osm_buildings(
            osm_data,
            airport["center_lat"],
            airport["center_lon"],
            airport["default_height_m"],
            airport["height_per_level_m"],
        )
        source_type = "osm"

    if not buildings:
        print("WARNING: No buildings found -> falling back to test cube.")
        create_test_cube(mesh_dir)
        return

    print(f"  Extruding {len(buildings)} polygons...")
    meshes = []
    building_records = []
    for polygon, height in buildings:
        mesh = extrude_polygon_to_mesh(polygon, height)
        if mesh is not None and len(mesh.faces) > 0:
            meshes.append(mesh)
            building_records.append(
                {
                    "height": float(height),
                    "coords": [[float(x), float(y)] for x, y in list(polygon.exterior.coords)],
                    "bounds": [float(v) for v in polygon.bounds],
                }
            )

    print(f"  {len(meshes)} meshes created.")
    if not meshes:
        print("ERROR: No valid meshes -> check polygon data.")
        return

    combined = trimesh.util.concatenate(meshes)
    airport_name = airport["name"]
    mesh_path = os.path.join(mesh_dir, f"{airport_name}_buildings.obj")
    combined.export(mesh_path)

    metadata = {
        "airport": airport_name,
        "source_type": source_type,
        "open_alaqs_database": openalaqs_db or None,
        "footprints_geojson": geojson_path or None,
        "n_buildings": len(meshes),
        "n_vertices": int(len(combined.vertices)),
        "n_faces": int(len(combined.faces)),
        "mesh_bbox_min": [float(v) for v in combined.bounds[0]],
        "mesh_bbox_max": [float(v) for v in combined.bounds[1]],
        "mesh_span_xyz_m": [float(v) for v in (combined.bounds[1] - combined.bounds[0])],
    }
    if openalaqs_context is not None:
        metadata["openalaqs_layer_counts"] = openalaqs_context["study"]["layer_counts"]
        metadata["openalaqs_reference_geojson"] = os.path.basename(openalaqs_context["files"]["geojson"])
        metadata["openalaqs_reference_summary"] = os.path.basename(openalaqs_context["files"]["summary"])

    with open(os.path.join(mesh_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(mesh_dir, "buildings_2d.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "airport": airport_name,
                "source_type": source_type,
                "n_buildings": len(building_records),
                "buildings": building_records,
            },
            f,
            indent=2,
        )

    print(f"  Saved -> {mesh_path}")
    print(
        f"  Stats: {metadata['n_buildings']} buildings, "
        f"{metadata['n_vertices']:,} vertices, {metadata['n_faces']:,} faces"
    )
    if openalaqs_context is not None:
        print(f"  OpenALAQS reference summary -> {openalaqs_context['files']['summary']}")
    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1: airport data ingestion and 3D extrusion")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--test-cube",
        action="store_true",
        help="Skip external data and generate a synthetic cube for quick testing",
    )
    args = parser.parse_args()
    run_ingestion(args.config, use_test_cube=args.test_cube)
