import streamlit as st
import pandas as pd
import openrouteservice
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium.plugins import MarkerCluster
import requests

# Main app
st.set_page_config(page_title="Logistics Buddy", page_icon="🚚")
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🚚 AI driven logistics agents")
with col2:
    #st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Wikipedia-logo.png", use_column_width=True)
    st.markdown("AI Agents create the most efficient delivery routes, detect disaster along the routes leveraging real time weather data and traffic conditions.")
tab1, tab2, tab3, tab4 = st.tabs(["Support Agent", "Route Planner Agent", "Disaster Management Agent", "Dashboard"])

st.set_page_config(page_title="Logistics Buddy", layout="wide")

with tab3:
    st.title("🚨 Disaster Management Agent")
    st.markdown("""
        Welcome to our AI-powered disaster management agent! I can help you with:
        - flood detection
        - Any other disaster detection
        """)

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

# Agent 1: Assistant Agent
with tab1:
    # Flow configuration
    #FLOW_ID = "YOUR_FLOW_ID_HERE"
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
            response = requests.request("POST", FLOW_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making API request: {e}")
        except ValueError as e:
            raise Exception(f"Error parsing response: {e}")

    # Configure the page
    st.set_page_config(
        page_title="Customer Support Agent",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for better styling
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

    # Title and description
    st.title("🤖  Support Agent")
    st.markdown("""
        Welcome to our AI-powered support agent! I can help you with:
        - Weather and traffic conditions
        - Recommendation for the best clothing
        - General FAQs
        """)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant message placeholder
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Run the flow with the user's message
                response = run_flow(
                    message=prompt,
                    output_type="chat",
                    input_type="chat",
                    tweaks=TWEAKS
                )

                # Extract the result from the response
                if isinstance(response, dict):
                    result = response['outputs'][0]['outputs'][0]['results']['message']['text']
                else:
                    result = response.get("result", "I apologize, but I couldn't process your request. Please try again.")
                
                message_placeholder.markdown(result)
                    
            except Exception as e:
                message_placeholder.markdown(f"I apologize, but I encountered an error: {str(e)}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": message_placeholder.markdown})

        
        # Add a clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun() 

# Agent 2: Route Planner Agent
with tab2:
    # Step 1: Araç sayısı
    st.subheader("Step 1: Select number of vehicles")
    num_vehicles = st.number_input(
        label="How many vehicles will be used for delivery?",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    # Step 2: Depo ve market datası
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

    df_markets = pd.DataFrame(market_data, columns=["Brand", "District", "Latitude", "Longitude"])
    df_markets["Label"] = df_markets["Brand"] + " - " + df_markets["District"]

    # Step 2: Depo seçimi
    st.subheader("Step 2: Select depot location")
    depot_label = st.selectbox(
        label="Choose the depot location (starting and ending point):",
        options=df_markets["Label"].unique()
    )

    # Step 3: Market seçimi
    st.subheader("Step 3: Select delivery markets")

    st.markdown("Tick the boxes for the delivery points you want to include in the route:")

    # Marketleri A-Z sırala
    sorted_market_labels = sorted(df_markets["Label"].unique())

    # Tıklanabilir kutular (checkbox) için boş liste oluştur
    selected_market_labels = []

    # Sütunlara bölerek daha kompakt liste yapalım (örneğin 3 sütun)
    cols = st.columns(3)

    for i, label in enumerate(sorted_market_labels):
        col = cols[i % 3]  # sırayla 3 kolona dağıt
        if col.checkbox(label, key=label):
            selected_market_labels.append(label)

    # Market listesi boşsa durdur
    if not selected_market_labels:
        st.warning("Please select at least one market to continue.")
        st.stop()

    # Depo market listesinde varsa çıkar
    if depot_label in selected_market_labels:
        selected_market_labels.remove(depot_label)

    # Seçilen verileri birleştir
    depot_df = df_markets[df_markets["Label"] == depot_label]
    markets_df = df_markets[df_markets["Label"].isin(selected_market_labels)]
    selected_markets = pd.concat([depot_df, markets_df]).drop_duplicates(subset="Label").reset_index(drop=True)
    selected_markets.loc[0, "Label"] = "📦 DEPO"


    # ORS_API_KEY = "5b3ce3597851110001cf624818bf8f0b6a2a41ddb5a91f9d31fc4131" 
    # Get your own API key from OpenRouteService
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

    # Veri modeli
    def create_data_model():
        data = {
            "distance_matrix": int_distance_matrix.tolist(),
            "num_vehicles": num_vehicles,
            "depot": 0
        }
        return data

    data = create_data_model()

    # OR-Tools model
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    # 1. Her markete gitmek zorunlu olsun (ceza)
    penalty = 100000
    for node in range(1, len(data["distance_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # 2. Araç başına kapasite kısıtı (en fazla market sayısı)
    demands = [0] + [1] * (len(data["distance_matrix"]) - 1)
    vehicle_capacities = [len(demands) // data["num_vehicles"]] * data["num_vehicles"]

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # capacity slack
        vehicle_capacities,
        True,
        'Capacity'
    )

    # Mesafe callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Arama parametreleri
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Çözüm bul
    solution = routing.SolveWithParameters(search_parameters)

    # Step 5: Rota çıktıları
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
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            node_index = manager.IndexToNode(index)
            route.append(selected_markets.loc[node_index, "Label"])

            st.markdown(f"### 🚛 Vehicle {vehicle_id + 1} Route:")
            st.write(" → ".join(route))
            st.write(f"🛣️ Distance: {route_distance / 1000:.2f} km")
            total_distance += route_distance

        st.markdown(f"### 📦 Total distance for all vehicles: **{total_distance / 1000:.2f} km**")
        # Harita oluştur (başlangıç noktası: depo)
        start_lat = selected_markets.iloc[0]["Latitude"]
        start_lon = selected_markets.iloc[0]["Longitude"]

        m = folium.Map(location=[start_lat, start_lon], zoom_start=11)
        colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen", "black", "pink"]
        marker_cluster = MarkerCluster().add_to(m)

        # Her araç için rota çiz
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            route_labels = []

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                lat = selected_markets.loc[node_index, "Latitude"]
                lon = selected_markets.loc[node_index, "Longitude"]
                label = selected_markets.loc[node_index, "Label"]
                route.append((lat, lon))
                route_labels.append(label)
                index = solution.Value(routing.NextVar(index))

            node_index = manager.IndexToNode(index)
            lat = selected_markets.loc[node_index, "Latitude"]
            lon = selected_markets.loc[node_index, "Longitude"]
            label = selected_markets.loc[node_index, "Label"]
            route.append((lat, lon))
            route_labels.append(label)

            folium.PolyLine(route, color=colors[vehicle_id % len(colors)],
                            weight=5, opacity=0.7,
                            tooltip=f"Vehicle {vehicle_id + 1}").add_to(m)

            for i, (coord, label) in enumerate(zip(route, route_labels)):
                folium.Marker(
                    location=coord,
                    popup=f"Vehicle {vehicle_id + 1} - Step {i + 1}: {label}",
                    icon=folium.Icon(color=colors[vehicle_id % len(colors)])
                ).add_to(marker_cluster)

        # Depo için özel ikon
        folium.Marker(
            location=[start_lat, start_lon],
            popup="📦 DEPO",
            icon=folium.Icon(color="gray", icon="home")
        ).add_to(m)

        # Haritayı Streamlit’te göster
        from streamlit_folium import st_folium

        st.subheader("Step 5: Route Map")
        st_data = st_folium(m, width=900, height=600)
    else:
        st.error("❌ No solution found. Please try with different settings.")

