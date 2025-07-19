import streamlit as st
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from route_analysis import get_distance_matrix, get_directions


def render(logistics_locations):
    """Render the route planner tab."""
    st.header("📍 Plan Your Delivery Route")

    # Create DataFrame from logistics locations
    df_locations = pd.DataFrame(logistics_locations, columns=[
        "Type", "Location", "Latitude", "Longitude"])
    df_locations["Label"] = df_locations["Type"] + \
        " - " + df_locations["Location"]

    # Input controls
    col1, col2 = st.columns(2)
    with col1:
        depot_label = st.selectbox(
            "Select your starting depot:",
            options=df_locations["Label"].unique(),
            index=2,  # Default to Doncaster for flood testing
            key="depot_selector"
        )

    with col2:
        num_vehicles = st.number_input(
            "Number of vehicles:", min_value=1, max_value=5, value=1)

    # Destination selection
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
        return

    if len(selected_destinations) == 1:
        st.info(
            "💡 Tip: Select multiple destinations for more efficient route optimization!")

    # Process route optimization
    route_locations = [depot_label] + selected_destinations
    selected_df = df_locations[df_locations["Label"].isin(route_locations)]
    selected_df = selected_df.reset_index(drop=True)
    selected_df.loc[0, "Label"] = "🏭 DEPOT"

    # Get distance matrix
    matrix_coords = selected_df[["Longitude", "Latitude"]].values.tolist()

    try:
        distance_matrix = get_distance_matrix(tuple(map(tuple, matrix_coords)))
    except Exception as e:
        st.error(f"Error calculating distances: {e}")
        return

    # Set up optimization problem
    data = {
        "distance_matrix": distance_matrix.tolist(),
        "num_vehicles": num_vehicles,
        "depot": 0
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    # Add constraints for multiple destinations
    if len(selected_destinations) > 1:
        penalty = 1000000
        for node in range(1, len(data["distance_matrix"])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Set up capacity constraints
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

    # Set distance cost
    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: data["distance_matrix"][manager.IndexToNode(
            from_index)][manager.IndexToNode(to_index)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 30

    with st.spinner("Optimizing routes..."):
        solution = routing.SolveWithParameters(search_parameters)

    st.subheader("🚛 Optimized Delivery Routes")

    if solution:
        # Create map
        route_map = folium.Map(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            zoom_start=10)

        colors = ["red", "blue", "green", "purple", "orange"]
        marker_cluster = MarkerCluster().add_to(route_map)
        total_distance = 0
        new_route_data = {}

        # Process each vehicle route
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route_display, route_coords, route_distance = [], [], 0

            # Build route
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_display.append(selected_df.loc[node_index, "Label"])
                route_coords.append(tuple(matrix_coords[node_index]))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            # Add final depot visit
            final_node_index = manager.IndexToNode(index)
            route_coords.append(tuple(matrix_coords[final_node_index]))

            # Only process routes with actual destinations
            if len(route_coords) > 2:
                new_route_data[f"vehicle_{vehicle_id}"] = {
                    "coords": route_coords,
                    "display": route_display,
                    "distance": route_distance
                }

                # Get route geometry and add to map
                try:
                    directions = get_directions(tuple(route_coords))
                    folium.GeoJson(
                        directions,
                        style_function=lambda x, color=colors[vehicle_id % len(colors)]: {
                            'color': color, 'weight': 5, 'opacity': 0.7
                        },
                        tooltip=f"Vehicle {vehicle_id + 1}"
                    ).add_to(route_map)
                except Exception as e:
                    st.warning(
                        f"Could not get route geometry for vehicle {vehicle_id + 1}: {e}")

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

        # Store route data for disaster management
        st.session_state.route_data = new_route_data

        # Display total distance
        if total_distance > 0:
            st.metric("Total Fleet Distance",
                      f"{total_distance / 1000:.1f} km")

        # Add depot marker
        folium.Marker(
            location=[selected_df.iloc[0]["Latitude"],
                      selected_df.iloc[0]["Longitude"]],
            popup="🏭 DEPOT",
            icon=folium.Icon(color="gray", icon="home")
        ).add_to(route_map)

        # Display map
        st_folium(route_map, width=900, height=500)

        # Success message for disaster management
        if new_route_data:
            st.success(
                "✅ Routes generated! You can now check for hazards in the Disaster Management tab.")

    else:
        st.error("❌ Could not generate optimal route.")
        if "route_data" in st.session_state:
            del st.session_state["route_data"]
