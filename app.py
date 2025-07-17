import streamlit as st
import pandas as pd
import openrouteservice
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium.plugins import MarkerCluster
import requests
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io
from streamlit_folium import st_folium
import pyproj

# --- Helper Functions for Disaster Management ---


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
                "properties": element.get('tags', {})
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
        if feature['geometry']['type'] == 'LineString':
            is_flooded = False
            is_near_flood = False
            for coord in feature['geometry']['coordinates']:
                # Transform road coordinate to the map's CRS
                lon, lat = transformer.transform(coord[0], coord[1])

                # Get pixel coordinates from geographic coordinates
                try:
                    py, px = rasterio.transform.rowcol(
                        flood_transform, lon, lat)
                except rasterio.errors.OutOfTransform:
                    continue

                # Check if the point is within the flood raster bounds
                if 0 <= px < flood_pixels.shape[1] and 0 <= py < flood_pixels.shape[0]:
                    if flood_pixels[py, px] == 1:
                        is_flooded = True
                        break

                    # Check a buffer around the point for nearby flooding
                    buffer = 3
                    min_x = max(0, px - buffer)
                    max_x = min(flood_pixels.shape[1], px + buffer + 1)
                    min_y = max(0, py - buffer)
                    max_y = min(flood_pixels.shape[0], py + buffer + 1)

                    if np.any(flood_pixels[min_y:max_y, min_x:max_x] == 1):
                        is_near_flood = True

            road_name = feature['properties'].get('name', 'Unnamed Road')
            if is_flooded:
                affected_roads.append(road_name)
            elif is_near_flood:
                near_flood_roads.append(road_name)

    return affected_roads, near_flood_roads


# Main app
st.set_page_config(page_title="Logistics Buddy", page_icon="🚚", layout="wide")
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🚚 AI driven logistics agents")
with col2:
    st.markdown("AI Agents create the most efficient delivery routes, detect disaster along the routes leveraging real time weather data and traffic conditions.")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Support Agent", "Route Planner Agent", "Disaster Management Agent", "Dashboard"])

# --- Tab 3: Disaster Management Agent ---
with tab3:
    st.title("🚨 Disaster Management Agent")
    st.markdown("Upload a GeoTIFF file to begin the flood analysis.")

    uploaded_file = st.file_uploader(
        "Upload a GeoTIFF file (.tif, .tiff)", type=["tif", "tiff"])

    if uploaded_file is not None:
        with st.spinner('Processing GeoTIFF and analyzing road impact...'):
            try:
                with rasterio.open(io.BytesIO(uploaded_file.read())) as src:
                    # Destination CRS for the map overlay
                    dst_crs = 'EPSG:3857'

                    # Calculate transform and dimensions for reprojecting to dst_crs
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds)

                    # Reproject the raster data
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

                    # Get bounds in the destination CRS (Web Mercator) for the overlay
                    dst_bounds = rasterio.transform.array_bounds(
                        height, width, transform)

                    # Get bounds in WGS 84 for centering the map and for API calls
                    wgs84_bounds = rasterio.warp.transform_bounds(
                        dst_crs, 'EPSG:4326', *dst_bounds)
                    west, south, east, north = wgs84_bounds

                    # CORRECTLY calculate map center using WGS 84 degree coordinates
                    map_center_lat = (south + north) / 2
                    map_center_lon = (west + east) / 2

                    # Create the Folium map
                    m = folium.Map(
                        location=[map_center_lat, map_center_lon], zoom_start=13)

                    # Create and add the flood overlay image
                    overlay_image = np.zeros(
                        (height, width, 4), dtype=np.uint8)
                    overlay_image[reprojected_data[0] == 1] = [
                        0, 100, 255, 150]  # RGBA for blue
                    folium.raster_layers.ImageOverlay(
                        image=overlay_image,
                        bounds=[[dst_bounds[1], dst_bounds[0]],
                                [dst_bounds[3], dst_bounds[2]]],
                        opacity=0.7,
                        name="Floodwater"
                    ).add_to(m)

                    # Get road data and analyze impact
                    road_data_overpass = get_road_data(
                        south, west, north, east)
                    road_data_geojson = overpass_to_geojson(road_data_overpass)
                    affected_roads, near_flood_roads = analyze_road_impact(
                        road_data_geojson,
                        reprojected_data[0],
                        transform,
                        dst_crs,
                        'EPSG:4326'  # GeoJSON is in WGS 84
                    )

                    # Add roads to map
                    if road_data_geojson['features']:
                        folium.GeoJson(
                            road_data_geojson,
                            name='OpenStreetMap Roads',
                            style_function=lambda x: {
                                'color': 'gray', 'weight': 2, 'opacity': 0.8}
                        ).add_to(m)

                    folium.LayerControl().add_to(m)

                st.success("Analysis complete!")
                st_folium(m, width=900, height=600)

                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.error(
                        f"**{len(affected_roads)} Roads Directly Flooded**")
                    # Display unique road names
                    st.write(list(set(affected_roads)))
                with col2:
                    st.warning(
                        f"**{len(near_flood_roads)} Roads Near Flooding**")
                    st.write(list(set(near_flood_roads)))

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

# --- Tab 4: Dashboard ---
with tab4:
    st.title("🚀 Dashboard")
    st.markdown("""
        Welcome to the AI-powered logistics tracking dashboard.
        """)

    # Sample metrics for overview
    metrics = {
        "Total Shipments": 500,
        "In Transit": 120,
        "Delivered": 350,
        "Pending": 30
    }
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments", metrics["Total Shipments"])
    col2.metric("In Transit", metrics["In Transit"])
    col3.metric("Delivered", metrics["Delivered"])
    col4.metric("Pending", metrics["Pending"])

# --- Tab 1: Support Agent ---
with tab1:
    FLOW_URL = f"http://localhost:7860/api/v1/run/customer-support2"
    TWEAKS = {}

    def run_flow(message, output_type="chat", input_type="chat", tweaks=None):
        """Run the Langflow flow with the given message."""
        payload = {
            "input_value": message,
            "output_type": output_type,
            "input_type": input_type
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.request(
                "POST", FLOW_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making API request: {e}")
        except ValueError as e:
            raise Exception(f"Error parsing response: {e}")

    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
        }
        .stMarkdown {
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stChatMessage[data-testid="stChatMessage"] {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🤖  Support Agent")
    st.markdown("""
        Welcome to our AI-powered support agent! I can help you with:
        - Weather and traffic conditions
        - Recommendation for the best clothing
        - General FAQs
        """)

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

            try:
                response = run_flow(
                    message=prompt,
                    output_type="chat",
                    input_type="chat",
                    tweaks=TWEAKS
                )

                if isinstance(response, dict):
                    result = response['outputs'][0]['outputs'][0]['results']['message']['text']
                else:
                    result = response.get(
                        "result", "I apologize, but I couldn't process your request. Please try again.")

                message_placeholder.markdown(result)

            except Exception as e:
                message_placeholder.markdown(
                    f"I apologize, but I encountered an error: {str(e)}")

        # Fixed to store the actual result
        st.session_state.messages.append(
            {"role": "assistant", "content": result})

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# --- Tab 2: Route Planner Agent ---
with tab2:
    st.subheader("Step 1: Select number of vehicles")
    num_vehicles = st.number_input(
        label="How many vehicles will be used for delivery?",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    market_data = [
        ("A101", "Kadıköy", 40.9850, 29.0273),
        ("A101", "Ümraniye", 41.0165, 29.1243),
        ("A101", "Beşiktaş", 41.0430, 29.0100),
        ("A101", "Bakırköy", 40.9777, 28.8723),
        ("A101", "Gaziosmanpaşa", 41.0611, 28.9155),
        ("Migros", "Kadıköy", 40.9900, 29.0305),
        ("Migros", "Şişli", 41.0595, 28.9872),
        ("Migros", "Bahçelievler", 41.0010, 28.8650),
        ("Migros", "Fatih", 41.0191, 28.9482),
        ("Şok", "Bağcılar", 41.0386, 28.8570),
        ("Şok", "Güngören", 41.0172, 28.8925),
        ("Şok", "Zeytinburnu", 41.0029, 28.9120),
        ("CarrefourSA", "Pendik", 40.8750, 29.2295),
        ("CarrefourSA", "Fatih", 41.0191, 28.9482),
        ("CarrefourSA", "Beşiktaş", 41.0428, 29.0083),
        ("Şok", "Tuzla", 40.8549, 29.3030),
    ]

    df_markets = pd.DataFrame(market_data, columns=[
                              "Brand", "District", "Latitude", "Longitude"])
    df_markets["Label"] = df_markets["Brand"] + " - " + df_markets["District"]

    st.subheader("Step 2: Select depot location")
    depot_label = st.selectbox(
        label="Choose the depot location (starting and ending point):",
        options=df_markets["Label"].unique()
    )

    st.subheader("Step 3: Select delivery markets")
    st.markdown(
        "Tick the boxes for the delivery points you want to include in the route:")
    sorted_market_labels = sorted(df_markets["Label"].unique())
    selected_market_labels = []
    cols = st.columns(3)

    for i, label in enumerate(sorted_market_labels):
        col = cols[i % 3]
        if col.checkbox(label, key=label):
            selected_market_labels.append(label)

    if not selected_market_labels:
        st.warning("Please select at least one market to continue.")
        st.stop()

    if depot_label in selected_market_labels:
        selected_market_labels.remove(depot_label)

    depot_df = df_markets[df_markets["Label"] == depot_label]
    markets_df = df_markets[df_markets["Label"].isin(selected_market_labels)]
    selected_markets = pd.concat([depot_df, markets_df]).drop_duplicates(
        subset="Label").reset_index(drop=True)
    selected_markets.loc[0, "Label"] = "📦 DEPO"

    ORS_API_KEY = st.secrets["ORS_API_KEY"]
    client = openrouteservice.Client(key=ORS_API_KEY)
    coordinates = selected_markets[["Longitude", "Latitude"]].values.tolist()

    matrix = client.distance_matrix(
        locations=coordinates,
        profile='driving-car',
        metrics=['distance'],
        units='km'
    )

    distance_matrix = matrix["distances"]
    int_distance_matrix = (np.array(distance_matrix) * 1000).astype(int)

    def create_data_model():
        data = {
            "distance_matrix": int_distance_matrix.tolist(),
            "num_vehicles": num_vehicles,
            "depot": 0
        }
        return data

    data = create_data_model()

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    penalty = 100000
    for node in range(1, len(data["distance_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    demands = [0] + [1] * (len(data["distance_matrix"]) - 1)
    vehicle_capacities = [len(demands) // data["num_vehicles"] + 1] * \
        data["num_vehicles"]  # Adjusted capacity

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vehicle_capacities,
        True,
        'Capacity'
    )

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    st.subheader("Step 4: Optimized Routes")
    if solution:
        total_distance = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(selected_markets.loc[node_index, "Label"])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            node_index = manager.IndexToNode(index)
            route.append(selected_markets.loc[node_index, "Label"])

            st.markdown(f"### 🚛 Vehicle {vehicle_id + 1} Route:")
            st.write(" → ".join(route))
            st.write(f"🛣️ Distance: {route_distance / 1000:.2f} km")
            total_distance += route_distance

        st.markdown(
            f"### 📦 Total distance for all vehicles: **{total_distance / 1000:.2f} km**")

        start_lat = selected_markets.iloc[0]["Latitude"]
        start_lon = selected_markets.iloc[0]["Longitude"]

        m = folium.Map(location=[start_lat, start_lon], zoom_start=11)
        colors = ["red", "blue", "green", "purple", "orange",
                  "darkred", "cadetblue", "darkgreen", "black", "pink"]
        marker_cluster = MarkerCluster().add_to(m)

        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route_coords = []
            route_labels = []

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                lat = selected_markets.loc[node_index, "Latitude"]
                lon = selected_markets.loc[node_index, "Longitude"]
                label = selected_markets.loc[node_index, "Label"]
                route_coords.append((lat, lon))
                route_labels.append(label)
                index = solution.Value(routing.NextVar(index))

            node_index = manager.IndexToNode(index)
            lat = selected_markets.loc[node_index, "Latitude"]
            lon = selected_markets.loc[node_index, "Longitude"]
            label = selected_markets.loc[node_index, "Label"]
            route_coords.append((lat, lon))
            route_labels.append(label)

            folium.PolyLine(route_coords, color=colors[vehicle_id % len(colors)],
                            weight=5, opacity=0.7,
                            tooltip=f"Vehicle {vehicle_id + 1}").add_to(m)

            for i, (coord, label) in enumerate(zip(route_coords, route_labels)):
                folium.Marker(
                    location=coord,
                    popup=f"Vehicle {vehicle_id + 1} - Step {i + 1}: {label}",
                    icon=folium.Icon(color=colors[vehicle_id % len(colors)])
                ).add_to(marker_cluster)

        folium.Marker(
            location=[start_lat, start_lon],
            popup="📦 DEPO",
            icon=folium.Icon(color="gray", icon="home")
        ).add_to(m)

        st.subheader("Step 5: Route Map")
        st_data = st_folium(m, width=900, height=600)
    else:
        st.error("❌ No solution found. Please try with different settings.")
