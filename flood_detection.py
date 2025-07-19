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
    url = "http://localhost:7860/api/v1/run/c496e528-0a6d-4be4-a4a7-f569309e1914"
    api_key = st.secrets.get("LANGFLOW_API_KEY", "")

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
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
    """Analyzes if route coordinates intersect with flooded pixels in a list of rasters."""
    if not list_of_flood_rasters:
        return []

    affected_segments = []

    for i, coord in enumerate(route_coords):
        lon, lat = coord[0], coord[1]

        for raster_data in list_of_flood_rasters:
            try:
                # Lazily create and cache transformers
                transformer_key = f"transformer_{raster_data['crs']}"
                if transformer_key not in st.session_state:
                    st.session_state[transformer_key] = pyproj.Transformer.from_crs(
                        'EPSG:4326', raster_data['crs'], always_xy=True
                    )
                transformer = st.session_state[transformer_key]

                # Check if coordinate is within the bounds of this tile
                min_tile_lon, min_tile_lat = raster_data['bounds'][0][1], raster_data['bounds'][0][0]
                max_tile_lon, max_tile_lat = raster_data['bounds'][1][1], raster_data['bounds'][1][0]

                if not (min_tile_lon <= lon <= max_tile_lon and min_tile_lat <= lat <= max_tile_lat):
                    continue

                # Transform coordinate to raster CRS
                lon_proj, lat_proj = transformer.transform(lon, lat)
                py, px = rasterio.transform.rowcol(
                    raster_data['transform'], lon_proj, lat_proj)

                # Check if coordinate intersects with flood pixel
                pixels = raster_data['pixels']
                if 0 <= py < pixels.shape[0] and 0 <= px < pixels.shape[1] and pixels[py, px] == 1:
                    affected_segments.append(
                        {"segment": i, "coordinate": coord})
                    break  # Move to next coordinate once flood is found

            except Exception as e:
                st.warning(f"Error checking coordinate {coord}: {e}")
                continue

    return affected_segments


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
