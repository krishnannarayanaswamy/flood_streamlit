import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium


def render(logistics_locations):
    """Render the fleet overview tab."""
    st.header("🚛 Fleet Management Overview")

    # Fleet metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Vehicles", "12", delta="2")
    col2.metric("Active Deliveries", "8", delta="-1")
    col3.metric("Fuel Costs (Today)", "£284", delta="+£45")
    col4.metric("Route Efficiency", "94%", delta="+2%")

    # Performance table
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

    # Service area map
    st.subheader("🌍 Service Area Coverage")
    df_locations = pd.DataFrame(logistics_locations, columns=[
        "Type", "Location", "Latitude", "Longitude"])

    coverage_map = folium.Map(location=[53.5, -1.2], zoom_start=8)
    for _, location in df_locations.iterrows():
        folium.Marker(
            location=[location["Latitude"], location["Longitude"]],
            popup=f"{location['Type']}<br>{location['Location']}",
            icon=folium.Icon(color='blue', icon='truck', prefix='fa')
        ).add_to(coverage_map)

    st_folium(coverage_map, width=900, height=400)
