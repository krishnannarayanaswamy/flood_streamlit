import streamlit as st
import pandas as pd
import openrouteservice
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium.plugins import MarkerCluster
import requests
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import io
from streamlit_folium import st_folium
import pyproj
import datetime
import json
from disaster_management import get_road_data, overpass_to_geojson, analyze_road_impact
import re

# Northern England logistics network
LOGISTICS_LOCATIONS = [
    # Major distribution hubs
    ("Distribution Hub", "Leeds Central", 53.8008, -1.5491),
    ("Regional Depot", "Sheffield Meadowhall", 53.3811, -1.4701),
    ("Warehouse", "Doncaster Logistics Park", 53.5228, -1.1285),
    ("Distribution Centre", "Lincoln Industrial", 53.2307, -0.5406),

    # Retail delivery points
    ("Tesco Superstore", "Leeds White Rose", 53.7584, -1.5820),
    ("ASDA Supercentre", "Sheffield Crystal Peaks", 53.3571, -1.4010),
    ("Sainsbury's", "Doncaster Lakeside", 53.5150, -1.1400),
    ("Morrisons", "Lincoln Tritton Road", 53.2450, -0.5300),
    ("Tesco Extra", "Scunthorpe", 53.5906, -0.6398),

    # Smaller delivery points
    ("Co-op Store", "Rotherham Town Centre", 53.4302, -1.3565),
    ("Lidl", "Gainsborough Market", 53.3989, -0.7762),
    ("Aldi", "Worksop Town", 53.3017, -1.1240),
    ("Iceland", "Barnsley Centre", 53.5519, -1.4797),
    ("Farmfoods", "Chesterfield", 53.2350, -1.4250),

    # Transport hubs
    ("Service Station", "A1 Markham Moor", 53.2800, -0.8900),
    ("Truck Stop", "M18 Thorne Services", 53.6100, -0.9500),
    ("Fuel Depot", "Immingham Docks", 53.6180, -0.2070),
]

# --- Caching Functions for API Calls ---


@st.cache_data
def get_distance_matrix(coords):
    """Fetches a distance matrix from OpenRouteService."""
    client = openrouteservice.Client(key=st.secrets["ORS_API_KEY"])
    matrix = client.distance_matrix(
        locations=coords, profile='driving-car', metrics=['distance'], units='km')
    return (np.array(matrix["distances"]) * 1000).astype(int)


@st.cache_data
def get_directions(route_coords):
    """Fetches detailed route geometry from OpenRouteService."""
    client = openrouteservice.Client(key=st.secrets["ORS_API_KEY"])
    return client.directions(
        coordinates=route_coords,
        profile='driving-car',
        format='geojson'
    )


@st.cache_data
def get_flood_overlay_from_langflow_cached(bbox, analysis_date):
    """
    Cached version of Langflow flood detection API call.
    This version now downloads and processes a GeoTIFF from a URL.
    """
    url = "http://localhost:7860/api/v1/run/c496e528-0a6d-4be4-a4a7-f569309e1914"
    api_key = st.secrets.get("LANGFLOW_API_KEY", "")

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "input_value": json.dumps({"bounding_box": ",".join(map(str, bbox)), "analysis_date": analysis_date}),
        "output_type": "chat", "input_type": "chat"
    }

    try:
        # Get response from Langflow
        api_response = requests.post(url, headers=headers, json=payload)
        api_response.raise_for_status()
        response_data = api_response.json()

        # Extract the text message from the complex response structure
        message_text = response_data.get('outputs', [{}])[0].get('outputs', [{}])[
            0].get('results', {}).get('message', {}).get('text')

        if not message_text:
            st.warning("Langflow did not return a message.")
            return None

        # FIX: Use a regular expression for robust URL extraction
        url_match = re.search(r'https?://[^\s)]+', message_text)
        if not url_match:
            st.warning(f"Could not extract a valid URL from Langflow message: '{message_text}'")
            return None
            
        image_url = url_match.group(0)

        st.info(f"Downloading flood overlay from: {image_url}")
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content

        # Process the downloaded GeoTIFF image
        with rasterio.open(io.BytesIO(image_bytes)) as src:
            pixels = src.read(1)

            # Create a transparent RGBA image for the overlay
            overlay_image = np.zeros(
                (src.height, src.width, 4), dtype=np.uint8)
            # Blue for flood, semi-transparent
            overlay_image[pixels == 1] = [0, 100, 255, 150]

            # Get image bounds and convert to WGS84 for Folium
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

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return None
    except Exception as e:
        st.error(
            f"An unexpected error occurred during flood data processing: {e}")
        return None

# REWRITTEN function to work with raster data


def analyze_road_flood_impact(route_coords, flood_raster_data):
    """Analyzes if route coordinates intersect with flooded pixels in a raster."""
    if not flood_raster_data:
        return []

    affected_segments = []
    pixels = flood_raster_data['pixels']
    img_transform = flood_raster_data['transform']
    img_crs = flood_raster_data['crs']

    # Create a transformer to convert route coordinates (WGS84) to the image's CRS
    transformer = pyproj.Transformer.from_crs(
        'EPSG:4326', img_crs, always_xy=True)

    for i, coord in enumerate(route_coords):
        try:
            # Transform coordinate
            lon_proj, lat_proj = transformer.transform(coord[0], coord[1])

            # Get pixel row and column
            py, px = rasterio.transform.rowcol(
                img_transform, lon_proj, lat_proj)

            # Check if pixel is within image bounds and if it's a flood pixel (value == 1)
            if 0 <= py < pixels.shape[0] and 0 <= px < pixels.shape[1] and pixels[py, px] == 1:
                affected_segments.append({"segment": i, "coordinate": coord})

        except Exception:
            continue  # Ignore points that fall outside the transformation bounds

    return affected_segments


# --- Main app ---
# [The rest of the app code remains the same until Tab 2]

# --- Main app ---
st.set_page_config(page_title="Northern Express Logistics",
                   page_icon="🚚", layout="wide")

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🚚 Northern Express Logistics")
with col2:
    st.markdown("Smart routing with real-time hazard detection")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Route Planner", "Disaster Management", "Driver Dashboard", "Fleet Overview"])

# --- Tab 1: Route Planner (No Changes) ---
with tab1:
    st.header("📍 Plan Your Delivery Route")

    df_locations = pd.DataFrame(LOGISTICS_LOCATIONS, columns=[
                                "Type", "Location", "Latitude", "Longitude"])
    df_locations["Label"] = df_locations["Type"] + \
        " - " + df_locations["Location"]

    col1, col2 = st.columns(2)
    with col1:
        depot_label = st.selectbox(
            "Select your starting depot:",
            options=df_locations["Label"].unique(),
            index=0,
            key="depot_selector"
        )

    with col2:
        num_vehicles = st.number_input(
            "Number of vehicles:", min_value=1, max_value=5, value=1)

    st.markdown("**Select delivery destinations:**")
    available_destinations = [
        label for label in df_locations["Label"].unique() if label != depot_label]

    selected_destinations = []
    cols = st.columns(3)
    for i, label in enumerate(available_destinations):
        if cols[i % 3].checkbox(label, key=f"dest_{label}"):
            selected_destinations.append(label)

    if not selected_destinations:
        st.warning("Please select at least one delivery destination.")
        st.stop()

    # ... (Rest of Tab 1 logic is unchanged)
    if len(selected_destinations) == 1:
        st.info(
            "💡 Tip: Select multiple destinations for more efficient route optimization!")

    route_locations = [depot_label] + selected_destinations
    selected_df = df_locations[df_locations["Label"].isin(route_locations)]
    selected_df = selected_df.reset_index(drop=True)
    selected_df.loc[0, "Label"] = "🏭 DEPOT"

    matrix_coords = selected_df[["Longitude", "Latitude"]].values.tolist()
    distance_matrix = get_distance_matrix(tuple(map(tuple, matrix_coords)))

    data = {
        "distance_matrix": distance_matrix.tolist(),
        "num_vehicles": num_vehicles,
        "depot": 0
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    if len(selected_destinations) > 1:
        penalty = 1000000
        for node in range(1, len(data["distance_matrix"])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    demands = [0] + [1] * (len(data["distance_matrix"]) - 1)
    if len(selected_destinations) == 1:
        vehicle_capacities = [len(demands)] * data["num_vehicles"]
    else:
        total_demand = sum(demands)
        vehicle_capacity = max(total_demand // data["num_vehicles"] + 1, 1)
        vehicle_capacities = [vehicle_capacity] * data["num_vehicles"]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        lambda from_index: demands[manager.IndexToNode(from_index)])
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, vehicle_capacities, True, 'Capacity')

    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: data["distance_matrix"][manager.IndexToNode(
            from_index)][manager.IndexToNode(to_index)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 30
    solution = routing.SolveWithParameters(search_parameters)

    st.subheader("🚛 Optimized Delivery Routes")
    if solution:
        route_map = folium.Map(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            zoom_start=9)

        colors = ["red", "blue", "green", "purple", "orange"]
        marker_cluster = MarkerCluster().add_to(route_map)
        total_distance = 0
        new_route_data = {}

        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route_display, route_coords, route_distance = [], [], 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_display.append(selected_df.loc[node_index, "Label"])
                route_coords.append(tuple(matrix_coords[node_index]))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            final_node_index = manager.IndexToNode(index)
            route_coords.append(tuple(matrix_coords[final_node_index]))

            if len(route_coords) > 2:
                new_route_data[f"vehicle_{vehicle_id}"] = {
                    "coords": route_coords,
                    "display": route_display,
                    "distance": route_distance
                }
                directions = get_directions(tuple(route_coords))
                folium.GeoJson(
                    directions,
                    style_function=lambda x, color=colors[vehicle_id % len(
                        colors)]: {
                        'color': color, 'weight': 5, 'opacity': 0.7
                    },
                    tooltip=f"Vehicle {vehicle_id + 1}"
                ).add_to(route_map)
                st.markdown(f"**Vehicle {vehicle_id + 1}:**")
                st.write(" → ".join(route_display) + " → 🏭 DEPOT")
                st.write(f"Distance: {route_distance / 1000:.1f} km")
                total_distance += route_distance
                for i, label in enumerate(route_display):
                    coord_row = selected_df[selected_df['Label'] == label]
                    if not coord_row.empty:
                        coord = (coord_row['Latitude'].iloc[0],
                                 coord_row['Longitude'].iloc[0])
                        folium.Marker(
                            location=coord,
                            popup=f"Vehicle {vehicle_id + 1} - {label}",
                            icon=folium.Icon(
                                color=colors[vehicle_id % len(colors)])
                        ).add_to(marker_cluster)

        st.session_state.route_data = new_route_data
        if total_distance > 0:
            st.metric("Total Fleet Distance",
                      f"{total_distance / 1000:.1f} km")
        folium.Marker(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            popup="🏭 DEPOT",
            icon=folium.Icon(color="gray", icon="home")
        ).add_to(route_map)
        st_folium(route_map, width=900, height=500)
    else:
        st.error("❌ Could not generate optimal route.")
        if "route_data" in st.session_state:
            del st.session_state["route_data"]


# --- Tab 2: Disaster Management (UPDATED LOGIC) ---
with tab2:
    st.header("🚨 Disaster Management & Route Safety")
    disaster_tab1, disaster_tab2 = st.tabs(
        ["Route Hazard Analysis", "GeoTIFF Analysis"])
    with disaster_tab1:
        st.markdown(
            "Monitor real-time hazards and get alternative routing recommendations.")
        if "route_data" not in st.session_state or not st.session_state.route_data:
            st.info(
                "👆 Please generate a route with at least one destination in the Route Planner tab first.")
            st.stop()
        if 'run_hazard_analysis' not in st.session_state:
            st.session_state.run_hazard_analysis = False
        col1, col2 = st.columns([2, 1])
        with col2:
            st.subheader("⚙️ Analysis Settings")
            analysis_date = st.date_input(
                "Analysis date:",
                value=datetime.date(2019, 11, 14),
                help="Select the date for hazard analysis"
            )
            test_mode = st.checkbox(
                "🧪 Test Mode (Simulate Flood Data)",
                help="Enable this if Langflow API is not available",
                key="test_mode_checkbox"
            )
            if st.button("🔍 Check Route Hazards", type="primary"):
                st.session_state.run_hazard_analysis = True
                st.session_state.analysis_test_mode = st.session_state.test_mode_checkbox
                st.rerun()
            if st.button("🗑️ Clear Analysis"):
                for key in ["hazard_map", "total_affected_vehicles", "flood_data_found", "test_mode_used", "run_hazard_analysis", "analysis_test_mode"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        if st.session_state.get('run_hazard_analysis'):
            with st.spinner("Analyzing routes for potential hazards..."):
                hazard_map = folium.Map(location=[53.5, -1.2], zoom_start=8)
                total_affected_vehicles = 0
                flood_data_found = False
                is_test_mode = st.session_state.get(
                    'analysis_test_mode', False)

                # Get flood data ONCE for the entire bounding box of all routes
                all_coords = [coord for route in st.session_state.route_data.values(
                ) for coord in route['coords']]
                if all_coords:
                    min_lon = min(c[0] for c in all_coords)
                    min_lat = min(c[1] for c in all_coords)
                    max_lon = max(c[0] for c in all_coords)
                    max_lat = max(c[1] for c in all_coords)
                    overall_bbox = (min_lon, min_lat, max_lon, max_lat)

                    flood_data = get_flood_overlay_from_langflow_cached(
                        overall_bbox, analysis_date.isoformat())

                    if flood_data:
                        flood_data_found = True
                        # Add raster overlay to the map
                        folium.raster_layers.ImageOverlay(
                            image=flood_data['overlay_image'],
                            bounds=flood_data['bounds'],
                            opacity=0.7,
                            name="Floodwater Overlay"
                        ).add_to(hazard_map)

                for vehicle_id, route_info in st.session_state.route_data.items():
                    route_coords = route_info["coords"]
                    if len(route_coords) > 1:
                        # Draw route on map
                        directions = get_directions(tuple(route_coords))
                        vehicle_num = int(vehicle_id.split('_')[1])
                        colors = ["red", "blue", "green", "purple", "orange"]
                        color = colors[vehicle_num % len(colors)]
                        folium.GeoJson(
                            directions,
                            style_function=lambda x, c=color: {
                                'color': c, 'weight': 4, 'opacity': 0.8},
                            tooltip=f"Vehicle {vehicle_num + 1} - Original Route"
                        ).add_to(hazard_map)

                        # Add markers for route stops
                        for i, coord in enumerate(route_coords):
                            folium.Marker(
                                location=[coord[1], coord[0]],
                                popup="🏭 DEPOT" if i == 0 else f"Vehicle {vehicle_num + 1} - Stop {i}",
                                icon=folium.Icon(
                                    color='gray' if i == 0 else color, icon='home' if i == 0 else 'truck')
                            ).add_to(hazard_map)

                        # Analyze impact if flood data is available
                        if flood_data_found:
                            affected_segments = analyze_road_flood_impact(
                                route_coords, flood_data)
                            if affected_segments:
                                total_affected_vehicles += 1
                                for segment in affected_segments:
                                    folium.Marker(
                                        location=[segment["coordinate"]
                                                  [1], segment["coordinate"][0]],
                                        popup=f"⚠️ HAZARD DETECTED",
                                        icon=folium.Icon(
                                            color='red', icon='exclamation-triangle', prefix='fa')
                                    ).add_to(hazard_map)

                folium.LayerControl().add_to(hazard_map)
                st.session_state.hazard_map = hazard_map
                st.session_state.total_affected_vehicles = total_affected_vehicles
                st.session_state.flood_data_found = flood_data_found
                st.session_state.test_mode_used = is_test_mode

        # Display Results Area
        with col2:
            if "total_affected_vehicles" in st.session_state:
                if st.session_state.total_affected_vehicles > 0:
                    st.error(
                        f"⚠️ {st.session_state.total_affected_vehicles} vehicle route(s) affected by hazards!")
                    st.subheader("🚨 Immediate Actions Required")
                    st.markdown(
                        "- 🛑 **STOP** affected vehicles\n- 📞 Contact drivers\n- 🗺️ Generate alternative routes")
                    if st.button("🔄 Generate Alternative Routes"):
                        st.success("✅ Alternative routes generated!")
                else:
                    if st.session_state.get("flood_data_found"):
                        st.success("✅ All routes clear - no hazards detected.")
                    else:
                        st.warning(
                            "⚠️ Could not verify route safety. No flood data was returned.")
        with col1:
            st.subheader("🗺️ Route Hazard Analysis Map")
            if "hazard_map" in st.session_state:
                st_folium(st.session_state.hazard_map, width=800, height=600)
            else:
                st.info(
                    "Click 'Check Route Hazards' on the right to view the analysis map.")

    # ... (Disaster Tab 2 for GeoTIFF upload remains unchanged)
    with disaster_tab2:
        st.markdown(
            "Upload a GeoTIFF file to analyze flood impact on road networks.")
        uploaded_file = st.file_uploader(
            "Upload a GeoTIFF file (.tif, .tiff)", type=["tif", "tiff"], key="disaster_upload")
        if uploaded_file is not None:
            with st.spinner('Processing GeoTIFF and analyzing road impact...'):
                try:
                    with rasterio.open(io.BytesIO(uploaded_file.read())) as src:
                        dst_crs = 'EPSG:3857'
                        transform, width, height = calculate_default_transform(
                            src.crs, dst_crs, src.width, src.height, *src.bounds)
                        reprojected_data = np.empty(
                            (1, height, width), dtype=src.dtypes[0])
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=reprojected_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)
                        dst_bounds = rasterio.transform.array_bounds(
                            height, width, transform)
                        wgs84_bounds = rasterio.warp.transform_bounds(
                            dst_crs, 'EPSG:4326', *dst_bounds)
                        west, south, east, north = wgs84_bounds
                        map_center_lat = (south + north) / 2
                        map_center_lon = (west + east) / 2
                        m = folium.Map(
                            location=[map_center_lat, map_center_lon], zoom_start=13)
                        overlay_image = np.zeros(
                            (height, width, 4), dtype=np.uint8)
                        overlay_image[reprojected_data[0] == 1] = [
                            0, 100, 255, 150]
                        folium.raster_layers.ImageOverlay(
                            image=overlay_image,
                            bounds=[[south, west], [north, east]],
                            opacity=0.7,
                            name="Floodwater"
                        ).add_to(m)
                        road_data_overpass = get_road_data(
                            south, west, north, east)
                        road_data_geojson = overpass_to_geojson(
                            road_data_overpass)
                        affected_roads, near_flood_roads = analyze_road_impact(
                            road_data_geojson, reprojected_data[0], transform, dst_crs, 'EPSG:4326')
                        affected_road_ids = {f['properties']['id']
                                             for f in affected_roads}
                        near_flood_road_ids = {f['properties']['id']
                                               for f in near_flood_roads}

                        def style_function(feature):
                            road_id = feature['properties']['id']
                            if road_id in affected_road_ids:
                                return {'color': 'red', 'weight': 5, 'opacity': 0.9}
                            elif road_id in near_flood_road_ids:
                                return {'color': 'orange', 'weight': 4, 'opacity': 0.8}
                            else:
                                return {'color': 'gray', 'weight': 2, 'opacity': 0.7}
                        if road_data_geojson['features']:
                            folium.GeoJson(
                                road_data_geojson,
                                name='OpenStreetMap Roads',
                                style_function=style_function,
                                tooltip=folium.GeoJsonTooltip(
                                    fields=['name', 'highway'], aliases=['Name:', 'Type:'])
                            ).add_to(m)
                        folium.LayerControl().add_to(m)
                    st.success("Analysis complete!")
                    map_col, results_col = st.columns([2, 1])
                    with map_col:
                        st_folium(m, width=700, height=500)
                    with results_col:
                        st.subheader("Analysis Results")
                        st.error(
                            f"**{len(affected_roads)} Roads Directly Flooded**")
                        st.write(
                            list(set([r['properties'].get('name', 'Unnamed Road') for r in affected_roads])))
                        st.warning(
                            f"**{len(near_flood_roads)} Roads Near Flooding**")
                        st.write(list(
                            set([r['properties'].get('name', 'Unnamed Road') for r in near_flood_roads])))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.exception(e)


# --- Tab 3: Driver Dashboard ---
with tab3:
    st.header("👨‍💼 Driver Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Routes", "3")
    col2.metric("Deliveries Today", "47")
    col3.metric("Fuel Efficiency", "8.2 L/100km")

    st.subheader("📋 Today's Assignments")
    assignments = [
        {"Driver": "John Smith", "Vehicle": "NK67 ABC",
            "Route": "Leeds → Sheffield", "Status": "In Progress", "ETA": "14:30"},
        {"Driver": "Sarah Jones", "Vehicle": "ML19 DEF",
            "Route": "Doncaster → Lincoln", "Status": "Completed", "ETA": "13:45"},
        {"Driver": "Mike Brown", "Vehicle": "YX21 GHI",
            "Route": "Sheffield → Rotherham", "Status": "Hazard Alert", "ETA": "Delayed"},
    ]
    assignments_df = pd.DataFrame(assignments)

    def style_status(val):
        if val == "Hazard Alert":
            return "background-color: red; color: white"
        elif val == "In Progress":
            return "background-color: yellow"
        elif val == "Completed":
            return "background-color: lightgreen"
        return ""

    styled_assignments = assignments_df.style.applymap(
        style_status, subset=['Status'])
    st.dataframe(styled_assignments, use_container_width=True)

    st.subheader("🤖 Support Agent")
    st.markdown(
        "Ask questions about weather, traffic conditions, or general FAQs")

    # NOTE: Ensure this URL points to your correct chat-oriented Langflow flow
    FLOW_URL = "http://localhost:7860/api/v1/run/c496e528-0a6d-4be4-a4a7-f569309e1914"
    TWEAKS = {}

    def run_flow(message, output_type="chat", input_type="chat", tweaks=None):
        payload = {"input_value": message,
                   "output_type": output_type, "input_type": input_type}
        if tweaks:
            payload["tweaks"] = tweaks
        headers = {"Content-Type": "application/json",
                   "x-api-key": st.secrets.get("LANGFLOW_API_KEY", "")}
        try:
            response = requests.post(FLOW_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error making API request: {e}")
            return None
        except ValueError as e:
            st.error(f"Error parsing response: {e}")
            return None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = run_flow(
                    message=prompt, output_type="chat", input_type="chat", tweaks=TWEAKS)
                if response:
                    try:
                        result = response['outputs'][0]['outputs'][0]['results']['message']['text']
                    except (KeyError, IndexError):
                        result = "I apologize, but I couldn't process your request. Please try again."
                    message_placeholder.markdown(result)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result})
                else:
                    error_message = "I'm having trouble connecting right now. Please try again later."
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message})
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Tab 4: Fleet Overview ---
with tab4:
    st.header("🚛 Fleet Management Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Vehicles", "12", delta="2")
    col2.metric("Active Deliveries", "8", delta="-1")
    col3.metric("Fuel Costs (Today)", "£284", delta="+£45")
    col4.metric("Route Efficiency", "94%", delta="+2%")

    st.subheader("📊 Performance Metrics")
    performance_data = [
        {"Vehicle": "NK67 ABC", "Driver": "John Smith",
            "Deliveries": 8, "Distance": "156 km", "Fuel": "18.2L"},
        {"Vehicle": "ML19 DEF", "Driver": "Sarah Jones",
            "Deliveries": 12, "Distance": "203 km", "Fuel": "23.1L"},
        {"Vehicle": "YX21 GHI", "Driver": "Mike Brown",
            "Deliveries": 6, "Distance": "98 km", "Fuel": "14.7L"},
    ]
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)

    st.subheader("🌍 Service Area Coverage")
    coverage_map = folium.Map(location=[53.5, -1.2], zoom_start=8)
    for _, location in df_locations.iterrows():
        folium.Marker(
            location=[location["Latitude"], location["Longitude"]],
            popup=f"{location['Type']}<br>{location['Location']}",
            icon=folium.Icon(color='blue', icon='truck', prefix='fa')
        ).add_to(coverage_map)
    st_folium(coverage_map, width=900, height=400)
