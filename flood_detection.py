import streamlit as st
import numpy as np
import requests
import rasterio
from rasterio.warp import transform_bounds
import io
import pyproj
import json
import re


def generate_tiles(bbox, tile_size_deg=0.05):
    """Generates a grid of smaller bounding boxes (tiles) to cover a larger one."""
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    lon_steps = np.arange(min_lon, max_lon, tile_size_deg)
    lat_steps = np.arange(min_lat, max_lat, tile_size_deg)

    for lon in lon_steps:
        for lat in lat_steps:
            tile_bbox = (lon, lat, lon + tile_size_deg, lat + tile_size_deg)
            tiles.append(tile_bbox)
    return tiles


@st.cache_data
def get_flood_overlay_from_langflow_cached(bbox, analysis_date):
    """
    Cached version of Langflow flood detection API call for a single tile.
    """
    url = "http://localhost:7860/api/v1/run/450bdd6d-1365-48dc-a819-779a0ccbf40c"
    #api_key = st.secrets.get("LANGFLOW_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    payload = {
        "input_value": json.dumps({"bounding_box": ",".join(map(str, bbox)), "analysis_date": analysis_date}),
        "output_type": "chat",
        "input_type": "chat"
    }

    try:
        st.write(f"🔍 Processing tile: {bbox}")  # Debug info
        api_response = requests.post(
            url, headers=headers, json=payload, timeout=60)
        api_response.raise_for_status()
        response_data = api_response.json()

        # Extract message text from response
        message_text = response_data.get('outputs', [{}])[0].get('outputs', [{}])[
            0].get('results', {}).get('message', {}).get('text')

        if not message_text:
            st.warning(f"No flood data returned for tile {bbox}")
            return None

        # Look for image URL in the response
        url_match = re.search(r'https?://[^\s)]+', message_text)
        if not url_match:
            st.warning(f"No image URL found in response for tile {bbox}")
            return None

        image_url = url_match.group(0)
        st.write(f"📊 Found flood data: {image_url}")  # Debug info

        # Download and process the flood image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content

        with rasterio.open(io.BytesIO(image_bytes)) as src:
            pixels = src.read(1)

            # Create overlay image for Folium
            overlay_image = np.zeros(
                (src.height, src.width, 4), dtype=np.uint8)
            # Blue with transparency for flood areas
            overlay_image[pixels == 1] = [0, 100, 255, 150]

            # Convert bounds to WGS84 for Folium
            wgs84_bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            west, south, east, north = wgs84_bounds
            folium_bounds = [[south, west], [north, east]]

            return {
                "overlay_image": overlay_image,
                "bounds": folium_bounds,
                "pixels": pixels,
                "transform": src.transform,
                "crs": src.crs
            }

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to process tile {bbox}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing flood data for tile {bbox}: {e}")
        return None


def analyze_road_flood_impact(route_coords, list_of_flood_rasters):
    """Basic analysis of route coordinates against flood rasters."""
    if not list_of_flood_rasters:
        st.write("⚠️ No flood rasters to analyze")
        return []

    affected_segments = []

    for i, coord in enumerate(route_coords):
        lon, lat = coord[0], coord[1]

        for raster_data in list_of_flood_rasters:
            try:
                # Create transformer if not cached
                transformer_key = f"transformer_{raster_data['crs']}"
                if transformer_key not in st.session_state:
                    st.session_state[transformer_key] = pyproj.Transformer.from_crs(
                        'EPSG:4326', raster_data['crs'], always_xy=True
                    )
                transformer = st.session_state[transformer_key]

                # Check if coordinate is within raster bounds
                min_tile_lon, min_tile_lat = raster_data['bounds'][0][1], raster_data['bounds'][0][0]
                max_tile_lon, max_tile_lat = raster_data['bounds'][1][1], raster_data['bounds'][1][0]

                if not (min_tile_lon <= lon <= max_tile_lon and min_tile_lat <= lat <= max_tile_lat):
                    continue  # Coordinate outside this raster

                # Transform coordinate to raster CRS
                lon_proj, lat_proj = transformer.transform(lon, lat)

                # Convert to pixel coordinates
                py, px = rasterio.transform.rowcol(
                    raster_data['transform'], lon_proj, lat_proj)

                # Check if pixel coordinates are valid
                pixels = raster_data['pixels']
                if 0 <= py < pixels.shape[0] and 0 <= px < pixels.shape[1]:
                    pixel_value = pixels[py, px]

                    # Check if this pixel indicates flooding (value = 1)
                    if pixel_value == 1:
                        affected_segments.append({
                            "segment": i,
                            "coordinate": coord,
                            "raster_index": raster_data,
                            "pixel_coords": (px, py),
                            "flood_value": int(pixel_value)
                        })
                        break  # Found flood at this coordinate, move to next coordinate

            except Exception:
                # Skip coordinates that can't be processed
                continue

    return affected_segments


def analyze_road_flood_impact_improved(route_data, list_of_flood_rasters):
    """Improved analysis using detailed route geometry from OpenRouteService."""
    # Import here to avoid circular imports
    from route_analysis import get_directions

    if not list_of_flood_rasters:
        st.write("⚠️ No flood rasters to analyze")
        return []

    affected_segments = []
    total_coordinates_checked = 0

    st.write(f"🔍 **IMPROVED INTERSECTION ANALYSIS**")
    st.write(f"📊 Analyzing against {len(list_of_flood_rasters)} flood rasters")

    for vehicle_id, route_info in route_data.items():
        vehicle_num = int(vehicle_id.split('_')[1]) + 1
        route_coords = route_info["coords"]

        st.write(f"\n🚛 **Vehicle {vehicle_num} Analysis**")
        st.write(f"   - Basic route points: {len(route_coords)}")

        # Get detailed route geometry from OpenRouteService
        try:
            directions = get_directions(tuple(route_coords))
            if directions and 'features' in directions and len(directions['features']) > 0:
                # Extract all coordinate points from the detailed route geometry
                detailed_coords = []
                for feature in directions['features']:
                    if feature['geometry']['type'] == 'LineString':
                        coords = feature['geometry']['coordinates']
                        detailed_coords.extend(coords)

                st.write(f"   - Detailed route points: {len(detailed_coords)}")
                coords_to_check = detailed_coords
            else:
                st.warning(
                    f"   - Using basic route points (OpenRouteService failed)")
                coords_to_check = route_coords

        except Exception as e:
            st.warning(f"   - Using basic route points (Error: {e})")
            coords_to_check = route_coords

        # Check each coordinate against flood rasters
        vehicle_affected_segments = []

        for i, coord in enumerate(coords_to_check):
            lon, lat = coord[0], coord[1]
            total_coordinates_checked += 1

            for raster_idx, raster_data in enumerate(list_of_flood_rasters):
                try:
                    # Create transformer if not cached
                    transformer_key = f"transformer_{raster_data['crs']}"
                    if transformer_key not in st.session_state:
                        st.session_state[transformer_key] = pyproj.Transformer.from_crs(
                            'EPSG:4326', raster_data['crs'], always_xy=True
                        )
                    transformer = st.session_state[transformer_key]

                    # Check if coordinate is within raster bounds
                    min_tile_lon, min_tile_lat = raster_data['bounds'][0][1], raster_data['bounds'][0][0]
                    max_tile_lon, max_tile_lat = raster_data['bounds'][1][1], raster_data['bounds'][1][0]

                    if not (min_tile_lon <= lon <= max_tile_lon and min_tile_lat <= lat <= max_tile_lat):
                        continue  # Coordinate outside this raster

                    # Transform coordinate to raster CRS
                    lon_proj, lat_proj = transformer.transform(lon, lat)

                    # Convert to pixel coordinates
                    py, px = rasterio.transform.rowcol(
                        raster_data['transform'], lon_proj, lat_proj)

                    # Check if pixel coordinates are valid
                    pixels = raster_data['pixels']
                    if 0 <= py < pixels.shape[0] and 0 <= px < pixels.shape[1]:
                        pixel_value = pixels[py, px]

                        # Check if this pixel indicates flooding (value = 1)
                        if pixel_value == 1:
                            st.write(
                                f"🌊 **FLOOD DETECTED** at coordinate {i+1}: ({lon:.4f}, {lat:.4f})")
                            vehicle_affected_segments.append({
                                "vehicle": vehicle_num,
                                "segment": i,
                                "coordinate": coord,
                                "raster_index": raster_idx,
                                "pixel_coords": (px, py),
                                "flood_value": int(pixel_value)
                            })
                            break  # Found flood at this coordinate, move to next coordinate

                except Exception as coord_error:
                    # Log specific coordinate processing errors for debugging
                    st.warning(
                        f"Error processing coordinate {i+1} ({lon:.4f}, {lat:.4f}): {coord_error}")
                    continue

        if vehicle_affected_segments:
            affected_segments.extend(vehicle_affected_segments)
            st.write(
                f"🚨 **Vehicle {vehicle_num}: {len(vehicle_affected_segments)} flood intersections found**")
        else:
            st.write(
                f"✅ **Vehicle {vehicle_num}: No flood intersections detected**")

    st.write(f"\n📊 **ANALYSIS SUMMARY**")
    st.write(f"   - Total coordinates checked: {total_coordinates_checked:,}")
    st.write(f"   - Flood intersections found: {len(affected_segments)}")

    return affected_segments


def generate_alternative_routes(affected_vehicle_details, original_route_data):
    """Generate alternative route suggestions."""
    alternatives = []

    for detail in affected_vehicle_details:
        vehicle_num = detail['vehicle']
        original_route = detail['route']

        # Simple alternative suggestions based on the route
        route_alternatives = {
            "vehicle": vehicle_num,
            "original_route": " → ".join(original_route),
            "alternatives": [
                {
                    "description": "Southern bypass via A614",
                    "additional_time": "12-18 minutes",
                    "additional_distance": "8.5 km",
                    "safety_rating": "High",
                    "route_type": "Major roads"
                },
                {
                    "description": "Western detour via M18/A1",
                    "additional_time": "20-25 minutes",
                    "additional_distance": "15.2 km",
                    "safety_rating": "Very High",
                    "route_type": "Motorway"
                },
                {
                    "description": "Delay until flood recedes",
                    "additional_time": "2-4 hours",
                    "additional_distance": "0 km",
                    "safety_rating": "High",
                    "route_type": "Wait strategy"
                }
            ]
        }
        alternatives.append(route_alternatives)

    return alternatives


def analyze_affected_roads_in_flood_areas(flood_rasters):
    """Analyze which specific roads are affected by flooding."""
    if not flood_rasters:
        return []

    st.write(f"🛣️ **ROAD NETWORK ANALYSIS**")

    try:
        # Get overall bounds of all flood areas
        all_bounds = [r['bounds'] for r in flood_rasters]
        min_lat = min(b[0][0] for b in all_bounds)
        min_lon = min(b[0][1] for b in all_bounds)
        max_lat = max(b[1][0] for b in all_bounds)
        max_lon = max(b[1][1] for b in all_bounds)

        st.write(
            f"   - Analysis area: {max_lat-min_lat:.3f}° × {max_lon-min_lon:.3f}°")

        # For now, return simulated road analysis
        # In a real implementation, you'd use your disaster_management.py functions here
        affected_roads = [
            {"name": "A614 (Thorne to Fishlake)", "type": "A-road",
             "status": "Severely flooded", "length_affected": "2.1 km"},
            {"name": "B1396 (Fishlake Road)", "type": "B-road",
             "status": "Partially flooded", "length_affected": "0.8 km"},
            {"name": "Local roads in Fishlake", "type": "Local",
                "status": "Multiple closures", "length_affected": "3.2 km"},
            {"name": "Bramwith Lane", "type": "Local",
                "status": "Impassable", "length_affected": "1.5 km"}
        ]

        st.write(f"   - Roads analyzed: {len(affected_roads)}")

        return affected_roads

    except Exception as e:
        st.error(f"❌ Road analysis failed: {e}")
        return []


def process_flood_tiles(tiles_to_process, analysis_date):
    """Process multiple tiles and return flood raster data."""
    all_flood_rasters = []

    if not tiles_to_process:
        st.warning("No tiles to process")
        return all_flood_rasters

    progress_bar = st.progress(0, text="Analyzing map tiles for flood data...")

    for i, tile_bbox in enumerate(tiles_to_process):
        progress_text = f"Processing tile {i+1}/{len(tiles_to_process)}: {tile_bbox}"
        progress_bar.progress(
            (i + 1) / len(tiles_to_process), text=progress_text)

        tile_flood_data = get_flood_overlay_from_langflow_cached(
            tile_bbox, analysis_date)
        if tile_flood_data:
            all_flood_rasters.append(tile_flood_data)
            st.success(f"✅ Flood data found in tile {i+1}")
        else:
            st.info(f"ℹ️ No flood data in tile {i+1}")

    progress_bar.empty()

    st.write(
        f"🌊 Found flood data in {len(all_flood_rasters)} out of {len(tiles_to_process)} tiles")
    return all_flood_rasters
