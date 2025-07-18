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
from route_analysis import get_distance_matrix, get_directions, get_flood_overlay_from_langflow

# Main app
st.set_page_config(page_title="Logistics Buddy", page_icon="🚚", layout="wide")
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🚚 AI driven logistics agents")
with col2:
    st.markdown(
        "AI Agents create the most efficient delivery routes, detect disaster along the routes leveraging real time weather data and traffic conditions.")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Support Agent", "Route Planner Agent", "Disaster Management Agent", "Dashboard"])

# --- Tab 3: Disaster Management Agent ---
with tab3:
    st.title("🚨 Disaster Management Agent")
    st.markdown("Upload a GeoTIFF file to begin the flood analysis.")

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

# --- Tab 4: Dashboard ---
with tab4:
    st.title("🚀 Dashboard")
    st.markdown("""
        Welcome to the AI-powered logistics tracking dashboard.
        """)
    metrics = {"Total Shipments": 500, "In Transit": 120,
               "Delivered": 350, "Pending": 30}
    col1_dash, col2_dash, col3_dash, col4_dash = st.columns(4)
    col1_dash.metric("Total Shipments", metrics["Total Shipments"])
    col2_dash.metric("In Transit", metrics["In Transit"])
    col3_dash.metric("Delivered", metrics["Delivered"])
    col4_dash.metric("Pending", metrics["Pending"])

# --- Tab 1: Support Agent ---
with tab1:
    st.title("🤖  Support Agent")
    st.markdown("""
        Welcome to our AI-powered support agent! I can help you with:
        - Weather and traffic conditions
        - Recommendation for the best clothing
        - General FAQs
        """)

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

# --- Tab 2: Route Planner Agent ---
with tab2:
    st.subheader("Step 1: Plan Your Route")
    num_vehicles = st.number_input(
        "How many vehicles will be used for delivery?", min_value=1, max_value=10, value=2, step=1)

    market_data = [("A101", "Kadıköy", 40.9850, 29.0273), ("A101", "Ümraniye", 41.0165, 29.1243), ("A101", "Beşiktaş", 41.0430, 29.0100), ("A101", "Bakırköy", 40.9777, 28.8723), ("A101", "Gaziosmanpaşa", 41.0611, 28.9155), ("Migros", "Kadıköy", 40.9900, 29.0305), ("Migros", "Şişli", 41.0595, 28.9872), ("Migros", "Bahçelievler", 41.0010, 28.8650),
                   ("Migros", "Fatih", 41.0191, 28.9482), ("Şok", "Bağcılar", 41.0386, 28.8570), ("Şok", "Güngören", 41.0172, 28.8925), ("Şok", "Zeytinburnu", 41.0029, 28.9120), ("CarrefourSA", "Pendik", 40.8750, 29.2295), ("CarrefourSA", "Fatih", 41.0191, 28.9482), ("CarrefourSA", "Beşiktaş", 41.0428, 29.0083), ("Şok", "Tuzla", 40.8549, 29.3030)]
    df_markets = pd.DataFrame(market_data, columns=[
                              "Brand", "District", "Latitude", "Longitude"])
    df_markets["Label"] = df_markets["Brand"] + " - " + df_markets["District"]

    depot_label = st.selectbox(
        "Choose the depot location (starting and ending point):", options=df_markets["Label"].unique())

    st.markdown(
        "Tick the boxes for the delivery points you want to include in the route:")
    sorted_market_labels = sorted(df_markets["Label"].unique())

    selected_market_labels = []
    cols = st.columns(3)
    for i, label in enumerate(sorted_market_labels):
        if cols[i % 3].checkbox(label, key=f"market_{label}"):
            selected_market_labels.append(label)

    delivery_locations = list(selected_market_labels)
    if depot_label in delivery_locations:
        delivery_locations.remove(depot_label)

    if not delivery_locations:
        st.warning(
            "Please select at least one delivery market that is not the depot.")
        st.stop()

    depot_df = df_markets[df_markets["Label"] == depot_label]
    markets_df = df_markets[df_markets["Label"].isin(
        delivery_locations)]
    selected_markets = pd.concat([depot_df, markets_df]).drop_duplicates(
        subset="Label").reset_index(drop=True)
    selected_markets.loc[0, "Label"] = "📦 DEPO"

    matrix_coords = selected_markets[["Longitude", "Latitude"]].values.tolist()

    distance_matrix = get_distance_matrix(tuple(map(tuple, matrix_coords)))

    data = {"distance_matrix": distance_matrix.tolist(
    ), "num_vehicles": num_vehicles, "depot": 0}
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    penalty = 100000
    for node in range(1, len(data["distance_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    demands = [0] + [1] * (len(data["distance_matrix"]) - 1)
    vehicle_capacities = [
        len(demands) // data["num_vehicles"]] * data["num_vehicles"]
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
    solution = routing.SolveWithParameters(search_parameters)

    st.subheader("Step 2: Review Route and Analyze for Floods")
    if solution:
        map_routes = folium.Map(location=[
                                selected_markets.iloc[0]["Latitude"], selected_markets.iloc[0]["Longitude"]], zoom_start=11)
        colors = ["red", "blue", "green", "purple", "orange",
                  "darkred", "cadetblue", "darkgreen", "black", "pink"]

        marker_cluster = MarkerCluster().add_to(map_routes)
        total_distance = 0

        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route_display, route_coords_for_api, route_distance = [], [], 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_display.append(selected_markets.loc[node_index, "Label"])
                route_coords_for_api.append(tuple(matrix_coords[node_index]))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            final_node_index = manager.IndexToNode(index)
            route_coords_for_api.append(tuple(matrix_coords[final_node_index]))

            if len(route_coords_for_api) > 1:
                directions = get_directions(tuple(route_coords_for_api))
                folium.GeoJson(
                    directions,
                    style_function=lambda x, color=colors[vehicle_id % len(colors)]: {
                        'color': color,
                        'weight': 5,
                        'opacity': 0.7
                    },
                    tooltip=f"Vehicle {vehicle_id + 1}"
                ).add_to(map_routes)

            st.markdown(f"### 🚛 Vehicle {vehicle_id + 1} Route:")
            st.write(" → ".join(route_display) + " → 📦 DEPO")
            st.write(f"🛣️ Distance: {route_distance / 1000:.2f} km")
            total_distance += route_distance

            for i, label in enumerate(route_display):
                coord = (selected_markets[selected_markets['Label'] == label]['Latitude'].iloc[0],
                         selected_markets[selected_markets['Label'] == label]['Longitude'].iloc[0])
                folium.Marker(
                    location=coord,
                    popup=f"Vehicle {vehicle_id + 1} - Stop {i + 1}: {label}",
                    icon=folium.Icon(color=colors[vehicle_id % len(colors)])
                ).add_to(marker_cluster)

        st.markdown(
            f"### 📦 Total distance for all vehicles: **{total_distance / 1000:.2f} km**")
        folium.Marker(location=[selected_markets.iloc[0]["Latitude"], selected_markets.iloc[0]["Longitude"]],
                      popup="📦 DEPO", icon=folium.Icon(color="gray", icon="home")).add_to(map_routes)

        folium.LayerControl().add_to(map_routes)

        # --- Flood Analysis UI ---
        st.markdown("---")
        st.subheader("Step 3: On-Demand Flood Analysis")

        analysis_date = st.date_input(
            "Select a date for analysis:", datetime.date.today())

        if st.button("Analyze Route for Flooding"):
            with st.spinner("Analyzing routes for potential flooding..."):
                all_directions = []
                for vehicle_id in range(data["num_vehicles"]):
                    index = routing.Start(vehicle_id)
                    route_coords_for_api = []
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        route_coords_for_api.append(
                            tuple(matrix_coords[node_index]))
                        index = solution.Value(routing.NextVar(index))
                    final_node_index = manager.IndexToNode(index)
                    route_coords_for_api.append(
                        tuple(matrix_coords[final_node_index]))

                    if len(route_coords_for_api) > 1:
                        all_directions.append(get_directions(
                            tuple(route_coords_for_api)))

                for i, directions in enumerate(all_directions):
                    bbox = tuple(directions['bbox'])

                    flood_overlay_geojson = get_flood_overlay_from_langflow(
                        bbox, analysis_date.isoformat())

                    if flood_overlay_geojson:
                        folium.GeoJson(
                            flood_overlay_geojson,
                            style_function=lambda x: {
                                'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.5, 'weight': 1},
                            name=f"Flood Overlay Vehicle {i+1}"
                        ).add_to(map_routes)

                st.success(
                    "Flood analysis complete. Check the map for flood overlays.")

        st_folium(map_routes, width=900, height=600, key="route_map")
    else:
        st.error("❌ No solution found. Please try with different settings.")
