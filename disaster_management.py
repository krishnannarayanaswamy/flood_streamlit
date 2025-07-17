import requests
import pyproj
import numpy as np
import rasterio


def get_road_data(south, west, north, east):
    """Fetches road data from OpenStreetMap via Overpass API."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
        way["highway"]
           ({south},{west},{north},{east});
    );
    out geom;
    """
    response = requests.post(overpass_url, data={"data": overpass_query})
    response.raise_for_status()
    return response.json()


def overpass_to_geojson(overpass_json):
    """Converts Overpass API JSON to a GeoJSON FeatureCollection."""
    features = []
    for element in overpass_json.get('elements', []):
        if element.get('type') == 'way' and 'geometry' in element:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[coord['lon'], coord['lat']] for coord in element['geometry']]
                },
                "properties": {**element.get('tags', {}), "id": element['id']}
            }
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def analyze_road_impact(roads_geojson, flood_pixels, flood_transform, map_crs, road_crs):
    """Identifies roads that are flooded or near a flood zone based on GeoJSON."""
    affected_roads = []
    near_flood_roads = []
    transformer = pyproj.Transformer.from_crs(
        road_crs, map_crs, always_xy=True)

    for feature in roads_geojson['features']:
        is_flooded = False
        is_near_flood = False
        for coord in feature['geometry']['coordinates']:
            lon, lat = transformer.transform(coord[0], coord[1])
            try:
                py, px = rasterio.transform.rowcol(flood_transform, lon, lat)
            except rasterio.errors.OutOfTransform:
                continue

            if 0 <= px < flood_pixels.shape[1] and 0 <= py < flood_pixels.shape[0]:
                if flood_pixels[py, px] == 1:
                    is_flooded = True
                    break

                buffer = 3
                min_x, max_x = max(
                    0, px - buffer), min(flood_pixels.shape[1], px + buffer + 1)
                min_y, max_y = max(
                    0, py - buffer), min(flood_pixels.shape[0], py + buffer + 1)

                if np.any(flood_pixels[min_y:max_y, min_x:max_x] == 1):
                    is_near_flood = True

        if is_flooded:
            affected_roads.append(feature)
        elif is_near_flood:
            near_flood_roads.append(feature)

    return affected_roads, near_flood_roads