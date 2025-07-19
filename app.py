from tabs import route_planner, disaster_management, driver_dashboard, fleet_overview
import streamlit as st


# Northern England logistics network (cleaned up - removed duplicates)
LOGISTICS_LOCATIONS = [
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
]

# --- Main App ---
st.set_page_config(page_title="Northern Express Logistics",
                   page_icon="🚚", layout="wide")

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🚚 Northern Express Logistics")
with col2:
    st.markdown("Smart routing with real-time hazard detection")

# Import tab modules

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Route Planner", "Disaster Management", "Driver Dashboard", "Fleet Overview"])

with tab1:
    route_planner.render(LOGISTICS_LOCATIONS)

with tab2:
    disaster_management.render()

with tab3:
    driver_dashboard.render()

with tab4:
    fleet_overview.render(LOGISTICS_LOCATIONS)
