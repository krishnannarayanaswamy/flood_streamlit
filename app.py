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
import datetime
import json
from disaster_management import get_road_data, overpass_to_geojson, analyze_road_impact

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
    """
    url = "http://localhost:7860/api/v1/run/c496e528-0a6d-4be4-a4a7-f569309e1914"
    api_key = st.secrets["LANGFLOW_API_KEY"]

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    # Create the data structure as expected by your Langflow component
    input_data = {
        # Convert bbox to comma-separated string
        "bounding_box": ",".join(map(str, bbox)),
        "analysis_date": analysis_date  # Send the date as an ISO-formatted string
    }

    # The input to your flow should be a JSON string of this data
    input_value_string = json.dumps(input_data)

    payload = {
        "input_value": input_value_string,
        "output_type": "chat",
        "input_type": "chat"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        # You will likely need to parse the actual GeoJSON from the response text
        response_data = response.json()
        if 'outputs' in response_data and response_data['outputs']:
            # Navigate through the nested structure to get the final message text
            message_text = response_data['outputs'][0]['outputs'][0]['results']['message']['text']
            # The text itself is a JSON string, so we need to parse it
            return json.loads(message_text)
        else:
            return None

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return None
    except json.JSONDecodeError as json_err:
        st.error(f"Failed to parse JSON from Langflow response: {json_err}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


def analyze_road_flood_impact(route_coords, flood_data):
    """Analyze if route intersects with flood areas"""
    if not flood_data or 'features' not in flood_data:
        return []

    affected_segments = []
    for i, coord in enumerate(route_coords):
        lon, lat = coord[0], coord[1]

        for feature in flood_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                # Simple bounding box check for flood intersection
                coords = feature['geometry']['coordinates'][0]
                min_lon = min(p[0] for p in coords)
                max_lon = max(p[0] for p in coords)
                min_lat = min(p[1] for p in coords)
                max_lat = max(p[1] for p in coords)

                if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                    affected_segments.append({
                        "segment": i,
                        "coordinate": coord,
                        "flood_area": feature.get('properties', {}).get('name', 'Flood Zone'),
                        "severity": feature.get('properties', {}).get('severity', 'unknown')
                    })

    return affected_segments


# Main app
st.set_page_config(page_title="Northern Express Logistics",
                   page_icon="🚚", layout="wide")

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🚚 Northern Express Logistics")
with col2:
    st.markdown("Smart routing with real-time hazard detection")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Route Planner", "Disaster Management", "Driver Dashboard", "Fleet Overview"])

# --- Tab 1: Route Planner ---
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
            key="depot_selector"  # Added unique key
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

    if len(selected_destinations) < 1:
        st.warning("Please select at least one delivery destination.")
        st.stop()

    # For better routing results, recommend multiple destinations
    if len(selected_destinations) == 1:
        st.info(
            "💡 Tip: Select multiple destinations for more efficient route optimization!")

    # Build route optimization
    route_locations = [depot_label] + selected_destinations
    selected_df = df_locations[df_locations["Label"].isin(route_locations)]
    selected_df = selected_df.reset_index(drop=True)
    selected_df.loc[0, "Label"] = "🏭 DEPOT"

    matrix_coords = selected_df[["Longitude", "Latitude"]].values.tolist()

    # Use cached distance matrix
    distance_matrix = get_distance_matrix(tuple(map(tuple, matrix_coords)))

    data = {
        "distance_matrix": distance_matrix.tolist(),
        "num_vehicles": num_vehicles,
        "depot": 0
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    # Add constraints - handle single vs multiple destinations differently
    if len(selected_destinations) == 1:
        # For single destination, don't allow skipping - make it mandatory
        # No disjunction constraints = all nodes must be visited
        pass
    else:
        # For multiple destinations, allow some flexibility with high penalty
        penalty = 1000000  # Much higher penalty to discourage skipping
        for node in range(1, len(data["distance_matrix"])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Simplified capacity constraints
    demands = [0] + [1] * (len(data["distance_matrix"]) - 1)

    # For single destination, ensure vehicle can handle it
    if len(selected_destinations) == 1:
        # Can handle all deliveries
        vehicle_capacities = [len(demands)] * data["num_vehicles"]
    else:
        # Distribute deliveries across vehicles
        total_demand = sum(demands)
        vehicle_capacity = max(total_demand // data["num_vehicles"] + 1, 1)
        vehicle_capacities = [vehicle_capacity] * data["num_vehicles"]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        lambda from_index: demands[manager.IndexToNode(from_index)])
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, vehicle_capacities, True, 'Capacity')

    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: data["distance_matrix"][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 30  # Add time limit
    solution = routing.SolveWithParameters(search_parameters)

    st.subheader("🚛 Optimized Delivery Routes")
    if solution:
        # Create map
        route_map = folium.Map(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            zoom_start=9)

        colors = ["red", "blue", "green", "purple", "orange"]
        marker_cluster = MarkerCluster().add_to(route_map)
        total_distance = 0

        # Store route data for disaster analysis
        if "route_data" not in st.session_state:
            st.session_state.route_data = {}
        st.session_state.route_data = {}

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

            # Store route for disaster analysis
            st.session_state.route_data[f"vehicle_{vehicle_id}"] = {
                "coords": route_coords,
                "display": route_display,
                "distance": route_distance
            }

            if len(route_coords) > 1:
                # Use cached directions
                directions = get_directions(tuple(route_coords))
                folium.GeoJson(
                    directions,
                    style_function=lambda x, color=colors[vehicle_id % len(colors)]: {
                        'color': color, 'weight': 5, 'opacity': 0.7
                    },
                    tooltip=f"Vehicle {vehicle_id + 1}"
                ).add_to(route_map)

            # Display route info
            st.markdown(f"**Vehicle {vehicle_id + 1}:**")
            st.write(" → ".join(route_display) + " → 🏭 DEPOT")
            st.write(f"Distance: {route_distance / 1000:.1f} km")
            total_distance += route_distance

            # Add markers
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

        st.metric("Total Fleet Distance", f"{total_distance / 1000:.1f} km")

        # Add depot marker
        folium.Marker(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            popup="🏭 DEPOT",
            icon=folium.Icon(color="gray", icon="home")
        ).add_to(route_map)

        st_folium(route_map, width=900, height=500)

    else:
        st.error("❌ Could not generate optimal route.")
        st.info(f"""
        **Debugging Info:**
        - Selected destinations: {len(selected_destinations)}
        - Total locations: {len(matrix_coords)}
        - Number of vehicles: {num_vehicles}
        
        **Suggestions:**
        - Try reducing the number of vehicles
        - Ensure API keys are configured correctly
        - Check that all locations are accessible by road
        """)

        # Still show a simple map with selected locations
        simple_map = folium.Map(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            zoom_start=9)

        for idx, row in selected_df.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=row["Label"],
                icon=folium.Icon(color="red" if idx == 0 else "blue")
            ).add_to(simple_map)

        st.subheader("📍 Selected Locations")
        st_folium(simple_map, width=900, height=500)

# --- Tab 2: Disaster Management ---
with tab2:
    st.header("🚨 Disaster Management & Route Safety")

    disaster_tab1, disaster_tab2 = st.tabs(
        ["Route Hazard Analysis", "GeoTIFF Analysis"])

    with disaster_tab1:
        st.markdown(
            "Monitor real-time hazards and get alternative routing recommendations.")

        if "route_data" not in st.session_state or not st.session_state.route_data:
            st.info("👆 Please generate a route in the Route Planner tab first.")
        else:
            col1, col2 = st.columns([2, 1])

            with col2:
                st.subheader("⚙️ Analysis Settings")
                analysis_date = st.date_input(
                    "Analysis date:",
                    # Default to a date when we know there was flooding
                    value=datetime.date(2019, 11, 14),
                    help="Select the date for hazard analysis"
                )

                if st.button("🔍 Check Route Hazards", type="primary"):
                    with st.spinner("Analyzing routes for potential hazards..."):
                        hazard_map = folium.Map(
                            location=[53.5, -1.2], zoom_start=8)

                        # Analyze each vehicle route
                        total_affected_vehicles = 0

                        for vehicle_id, route_info in st.session_state.route_data.items():
                            route_coords = route_info["coords"]

                            if len(route_coords) > 1:
                                # Get directions for detailed route (cached)
                                directions = get_directions(
                                    tuple(route_coords))
                                bbox = tuple(directions['bbox'])

                                # Call Langflow for flood detection (cached)
                                flood_overlay = get_flood_overlay_from_langflow_cached(
                                    bbox, analysis_date.isoformat())

                                # Add original route to map
                                vehicle_num = int(vehicle_id.split('_')[1])
                                colors = ["red", "blue",
                                          "green", "purple", "orange"]
                                color = colors[vehicle_num % len(colors)]

                                folium.GeoJson(
                                    directions,
                                    style_function=lambda x, c=color: {
                                        'color': c, 'weight': 4, 'opacity': 0.7},
                                    tooltip=f"Vehicle {vehicle_num + 1} - Original Route"
                                ).add_to(hazard_map)

                                # Add flood overlay if detected
                                if flood_overlay:
                                    folium.GeoJson(
                                        flood_overlay,
                                        style_function=lambda x: {
                                            'color': 'blue', 'fillColor': 'blue',
                                            'fillOpacity': 0.5, 'weight': 2
                                        },
                                        name=f"Flood Risk - Vehicle {vehicle_num + 1}"
                                    ).add_to(hazard_map)

                                    # Analyze impact
                                    affected_segments = analyze_road_flood_impact(
                                        route_coords, flood_overlay)

                                    if affected_segments:
                                        total_affected_vehicles += 1

                                        # Add warning markers
                                        for segment in affected_segments:
                                            coord = segment["coordinate"]
                                            folium.Marker(
                                                location=[coord[1], coord[0]],
                                                popup=f"⚠️ HAZARD DETECTED<br>Vehicle {vehicle_num + 1}<br>Flood Risk Area",
                                                icon=folium.Icon(
                                                    color='red', icon='exclamation-triangle', prefix='fa')
                                            ).add_to(hazard_map)

                        folium.LayerControl().add_to(hazard_map)

                        # Store map in session state so it persists
                        st.session_state.hazard_map = hazard_map
                        st.session_state.total_affected_vehicles = total_affected_vehicles

                # Add clear button
                if "hazard_map" in st.session_state:
                    if st.button("🗑️ Clear Analysis"):
                        if "hazard_map" in st.session_state:
                            del st.session_state.hazard_map
                        if "total_affected_vehicles" in st.session_state:
                            del st.session_state.total_affected_vehicles
                        st.rerun()

                # Display results outside the button handler
                if "total_affected_vehicles" in st.session_state:
                    if st.session_state.total_affected_vehicles > 0:
                        st.error(
                            f"⚠️ {st.session_state.total_affected_vehicles} vehicle route(s) affected by hazards!")

                        st.subheader("🚨 Immediate Actions Required")
                        st.markdown("""
                        **High Priority:**
                        - 🛑 **STOP** affected vehicles immediately
                        - 📞 Contact drivers on affected routes
                        - 🗺️ Generate alternative routes below
                        
                        **Next Steps:**
                        - ⏱️ Expect 30-60 minute delays
                        - 💰 Additional fuel costs for longer routes
                        - 📱 Update customers with new ETAs
                        """)

                        if st.button("🔄 Generate Alternative Routes"):
                            st.success(
                                "✅ Alternative routes generated! Check with fleet manager for approval.")
                    else:
                        st.success(
                            "✅ All routes clear - no hazards detected")
                        st.info("Safe to proceed with planned routes")

            with col1:
                if "hazard_map" in st.session_state:
                    st.subheader("🗺️ Route Hazard Analysis")
                    st_folium(st.session_state.hazard_map,
                              width=600, height=500)
                else:
                    st.info("Click 'Check Route Hazards' to see analysis results")

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

    FLOW_URL = f"http://localhost:7860/api/v1/run/customer-support2"
    TWEAKS = {}

    def run_flow(message, output_type="chat", input_type="chat", tweaks=None):
        payload = {"input_value": message,
                   "output_type": output_type, "input_type": input_type}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.request(
                "POST", FLOW_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making API request: {e}")
        except ValueError as e:
            raise Exception(f"Error parsing response: {e}")

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
                    message=prompt, output_type="chat", input_type="chat", tweaks=TWEAKS)
                result = response.get('outputs', [{}])[0].get('outputs', [{}])[0].get('results', {}).get(
                    'message', {}).get('text', "I apologize, but I couldn't process your request. Please try again.")
                message_placeholder.markdown(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result})
            except Exception as e:
                error_message = f"I apologize, but I encountered an error: {str(e)}"
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

    # Sample performance data
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

    # Simple coverage map
    coverage_map = folium.Map(location=[53.5, -1.2], zoom_start=8)

    # Add all logistics locations
    for _, location in pd.DataFrame(LOGISTICS_LOCATIONS, columns=["Type", "Location", "Lat", "Lon"]).iterrows():
        folium.Marker(
            location=[location["Lat"], location["Lon"]],
            popup=f"{location['Type']}<br>{location['Location']}",
            icon=folium.Icon(color='blue', icon='truck', prefix='fa')
        ).add_to(coverage_map)

    st_folium(coverage_map, width=900, height=400)
