import streamlit as st
import pandas as pd
import openrouteservice
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium.plugins import MarkerCluster
import requests
from streamlit_folium import st_folium

# Import your custom modules
try:
    from tabs import route_planner, disaster_management, driver_dashboard, fleet_overview
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    st.warning("Some advanced modules not available. Using basic functionality.")

# Main app configuration
st.set_page_config(page_title="🚚 AI Logistics Buddy",
                   page_icon="🚚", layout="wide")

# Header
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🚚 AI-Driven Logistics Agents")
with col2:
    st.markdown(
        "**Smart routing • Real-time hazard detection • AI-powered support**")

# Navigation tabs
if MODULES_AVAILABLE:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Support Agent",
        "🗺️ Route Planner",
        "🚨 Disaster Management",
        "👨‍💼 Driver Dashboard",
        "📊 Fleet Overview"
    ])
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Support Agent",
        "🗺️ Basic Route Planner",
        "🚨 Basic Disaster Management",
        "📊 Dashboard"
    ])

# === TAB 1: SUPPORT AGENT ===
with tab1:
    st.header("🤖 AI Support Agent")
    st.markdown("""
    Welcome to our AI-powered support agent! I can help you with:
    - 🌤️ Weather and traffic conditions
    - 👔 Recommendations for optimal delivery times
    - ❓ General logistics FAQs
    - 🚛 Fleet management questions
    """)

    # Flow configuration
    FLOW_URL = "http://localhost:7860/api/v1/run/customer-support2"

    def run_flow(message, output_type="chat", input_type="chat", tweaks=None):
        """Run the Langflow flow with the given message."""
        payload = {
            "input_value": message,
            "output_type": output_type,
            "input_type": input_type
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": st.secrets.get("LANGFLOW_API_KEY", "")
        }

        try:
            response = requests.post(
                FLOW_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            return None
        except ValueError as e:
            st.error(f"Response parsing error: {e}")
            return None

    # Custom CSS for better chat styling
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today?", key="support_agent_chat"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner("Thinking..."):
                try:
                    response = run_flow(
                        message=prompt, output_type="chat", input_type="chat")

                    if response and 'outputs' in response:
                        result = response['outputs'][0]['outputs'][0]['results']['message']['text']
                    else:
                        result = "I apologize, but I couldn't process your request. Please try again."

                    message_placeholder.markdown(result)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result})

                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg})

    # Clear chat button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# === TAB 2: ROUTE PLANNER ===
with tab2:
    if MODULES_AVAILABLE:
        # Use advanced route planner
        route_planner.render([
            # Major distribution hubs
            ("Distribution Hub", "Leeds Central", 53.8008, -1.5491),
            ("Regional Depot", "Sheffield Meadowhall", 53.3811, -1.4701),
            ("Warehouse", "Doncaster Logistics Park", 53.5228, -1.1285),
            ("Distribution Centre", "Lincoln Industrial", 53.2307, -0.5406),
            ("Distribution Point", "Askern North", 53.6150, -1.1500),

            # Retail delivery points
            ("Tesco Superstore", "Leeds White Rose", 53.7584, -1.5820),
            ("ASDA Supercentre", "Sheffield Crystal Peaks", 53.3571, -1.4010),
            ("Sainsbury's", "Doncaster Lakeside", 53.5150, -1.1400),
            ("Morrisons", "Lincoln Tritton Road", 53.2450, -0.5300),
            ("Tesco Extra", "Scunthorpe", 53.5906, -0.6398),
            ("Delivery Point", "Fishlake Village", 53.612819, -1.014153),

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
        ])
    else:
        # Basic route planner (your existing code)
        st.header("🗺️ Route Planner Agent")

        # Your existing route planner code
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
            ("Migros", "Kadıköy", 40.9900, 29.0305),
            ("Migros", "Şişli", 41.0595, 28.9872),
            ("Şok", "Bağcılar", 41.0386, 28.8570),
        ]

        df_markets = pd.DataFrame(market_data, columns=[
                                  "Brand", "District", "Latitude", "Longitude"])
        df_markets["Label"] = df_markets["Brand"] + \
            " - " + df_markets["District"]

        depot_label = st.selectbox(
            label="Choose the depot location:",
            options=df_markets["Label"].unique()
        )

        st.markdown("**Select delivery markets:**")
        selected_market_labels = []
        cols = st.columns(3)

        for i, label in enumerate(df_markets["Label"].unique()):
            if label != depot_label:
                col = cols[i % 3]
                if col.checkbox(label, key=f"market_{label}"):
                    selected_market_labels.append(label)

        if selected_market_labels and st.button("🚛 Generate Routes"):
            st.success(
                f"✅ Route generated for {len(selected_market_labels)} destinations!")
            st.info(
                "💡 Upgrade to advanced modules for flood detection and detailed route optimization.")

# === TAB 3: DISASTER MANAGEMENT ===
with tab3:
    if MODULES_AVAILABLE:
        # Use advanced disaster management
        disaster_management.render()
    else:
        # Basic disaster management
        st.header("🚨 Disaster Management Agent")
        st.markdown("""
        Welcome to our AI-powered disaster management agent! I can help you with:
        - 🌊 Flood detection
        - ⛈️ Weather monitoring
        - 🚨 Route safety analysis
        """)

        st.info("💡 Connect your route data to enable real-time hazard detection.")

        # Simple disaster input
        disaster_type = st.selectbox("Select disaster type:", [
                                     "Flood", "Storm", "Traffic Incident", "Road Closure"])
        location = st.text_input("Enter location:")
        severity = st.slider("Severity level:", 1, 5, 3)

        if st.button("🚨 Report Incident"):
            st.warning(
                f"⚠️ {disaster_type} reported at {location} with severity level {severity}")
            st.info(
                "Upgrade to advanced modules for automated satellite-based flood detection.")

# === TAB 4: DRIVER DASHBOARD / FLEET OVERVIEW ===
if MODULES_AVAILABLE:
    with tab4:
        driver_dashboard.render()

    with tab5:
        fleet_overview.render([
            ("Distribution Hub", "Leeds Central", 53.8008, -1.5491),
            ("Warehouse", "Doncaster Logistics Park", 53.5228, -1.1285),
            ("Sainsbury's", "Doncaster Lakeside", 53.5150, -1.1400),
        ])
else:
    with tab4:
        st.header("📊 Fleet Dashboard")

        # Sample metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Shipments", "500", "+50")
        col2.metric("In Transit", "120", "+12")
        col3.metric("Delivered", "350", "+38")
        col4.metric("Pending", "30", "-8")

        # Sample chart
        st.subheader("📈 Daily Performance")
        chart_data = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            'Deliveries': [45, 52, 48, 61, 55],
            'Delays': [3, 1, 2, 0, 2]
        })
        st.bar_chart(chart_data.set_index('Day'))

        # Fleet status
        st.subheader("🚛 Fleet Status")
        fleet_status = [
            {"Vehicle": "Vehicle 001", "Status": "En Route",
                "Location": "Doncaster", "ETA": "14:30"},
            {"Vehicle": "Vehicle 002", "Status": "Loading",
                "Location": "Depot", "ETA": "15:45"},
            {"Vehicle": "Vehicle 003", "Status": "Delivered",
                "Location": "Leeds", "ETA": "Completed"},
        ]

        for status in fleet_status:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            col1.write(f"🚚 {status['Vehicle']}")

            if status['Status'] == 'En Route':
                col2.success(f"✅ {status['Status']}")
            elif status['Status'] == 'Loading':
                col2.warning(f"⏳ {status['Status']}")
            else:
                col2.info(f"🏁 {status['Status']}")

            col3.write(f"📍 {status['Location']}")
            col4.write(f"⏰ {status['ETA']}")

# === FOOTER ===
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🚚 AI Logistics Buddy • Powered by Machine Learning • Real-time Intelligence
    </div>
    """, unsafe_allow_html=True)

# === SIDEBAR (Optional) ===
with st.sidebar:
    st.markdown("### 🚚 Quick Actions")

    if st.button("📞 Emergency Contact", use_container_width=True):
        st.success("🚨 Emergency services contacted!")

    if st.button("📍 Track All Vehicles", use_container_width=True):
        st.info("🚛 Tracking 3 active vehicles...")

    if st.button("🌤️ Weather Update", use_container_width=True):
        st.info("☀️ Clear conditions. Safe for deliveries.")

    st.markdown("### 📊 Quick Stats")
    st.metric("Active Routes", "3")
    st.metric("Avg. Delivery Time", "28 min")
    st.metric("Fuel Efficiency", "8.2 L/100km")

    if not MODULES_AVAILABLE:
        st.markdown("### 🔧 Upgrade")
        st.warning("Install advanced modules for full functionality:")
        st.code("pip install -r requirements.txt")
