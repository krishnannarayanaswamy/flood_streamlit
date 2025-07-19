import streamlit as st
import folium
from streamlit_folium import st_folium
import datetime
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io
from disaster_management import get_road_data, overpass_to_geojson, analyze_road_impact
from flood_detection import generate_tiles, process_flood_tiles, analyze_road_flood_impact
from route_analysis import get_directions


def render_tile_configuration():
    """Render the tile configuration section."""
    st.subheader("🔧 Tile Configuration")

    tile_size = st.select_slider(
        "Tile size (degrees):",
        options=[0.01, 0.025, 0.05, 0.1, 0.2],
        value=0.1,  # Default to faster processing
        format_func=lambda x: f"{x}° (~{x * 111:.1f} km)",
        help="Smaller tiles = more detailed analysis but more API calls"
    )

    return tile_size


def render_tile_metrics(current_bbox, tile_size):
    """Render tile count and metrics."""
    estimated_tiles = generate_tiles(current_bbox, tile_size)

    col_tiles1, col_tiles2, col_tiles3 = st.columns(3)

    with col_tiles1:
        st.metric(
            "Tiles to Process",
            len(estimated_tiles),
            help="Number of API calls needed for flood detection"
        )

    with col_tiles2:
        est_time_min = len(estimated_tiles) * 2
        est_time_max = len(estimated_tiles) * 5
        st.metric(
            "Est. Processing Time",
            f"{est_time_min}-{est_time_max}s",
            help="Estimated time for all API calls"
        )

    with col_tiles3:
        bbox_span_km = ((current_bbox[2] - current_bbox[0]) * 111,
                        (current_bbox[3] - current_bbox[1]) * 111)
        st.metric(
            "Coverage Area",
            f"{bbox_span_km[0]:.0f}×{bbox_span_km[1]:.0f} km",
            help="Geographic area to be analyzed"
        )

    # Warnings for large tile counts
    if len(estimated_tiles) > 100:
        st.warning(
            f"⚠️ {len(estimated_tiles)} tiles will require many API calls. Consider using a larger tile size.")
    elif len(estimated_tiles) > 50:
        st.info(
            f"ℹ️ {len(estimated_tiles)} tiles selected. Processing may take up to {est_time_max} seconds.")
    else:
        st.success(
            f"✅ {len(estimated_tiles)} tiles - good for real-time analysis!")

    # Detailed breakdown in expander
    with st.expander("📊 Detailed Tile Analysis"):
        st.write(f"**Bounding Box:** {current_bbox}")
        st.write(
            f"**Longitude span:** {current_bbox[2] - current_bbox[0]:.3f}° ({(current_bbox[2] - current_bbox[0]) * 111:.1f} km)")
        st.write(
            f"**Latitude span:** {current_bbox[3] - current_bbox[1]:.3f}° ({(current_bbox[3] - current_bbox[1]) * 111:.1f} km)")
        st.write(
            f"**Grid dimensions:** {int(np.ceil((current_bbox[2] - current_bbox[0]) / tile_size))} × {int(np.ceil((current_bbox[3] - current_bbox[1]) / tile_size))}")
        st.write(
            f"**Tile coverage:** {tile_size * 111:.1f} km × {tile_size * 111:.1f} km each")

    return estimated_tiles


def perform_hazard_analysis(route_data, tile_size, analysis_date):
    """Performs data analysis for hazards, returns data not a map."""
    with st.spinner("🔍 Analyzing route areas for potential hazards..."):
        all_coords = [coord for route in route_data.values() for coord in route['coords']]
        overall_bbox = (
            min(c[0] for c in all_coords) - 0.01, min(c[1] for c in all_coords) - 0.01,
            max(c[0] for c in all_coords) + 0.01, max(c[1] for c in all_coords) + 0.01
        )
        st.write(f"🗺️ Analysis area: {overall_bbox}")

        tiles_to_process = generate_tiles(overall_bbox, tile_size)
        st.write(f"📊 Processing {len(tiles_to_process)} tiles...")

        all_flood_rasters = process_flood_tiles(tiles_to_process, analysis_date.isoformat())

        total_affected_vehicles = 0
        for vehicle_id, route_info in route_data.items():
            affected_segments = analyze_road_flood_impact(route_info["coords"], all_flood_rasters)
            if affected_segments:
                total_affected_vehicles += 1
        
        return total_affected_vehicles, bool(all_flood_rasters), all_flood_rasters


def create_hazard_map(route_data, all_flood_rasters):
    """Creates a Folium map with route and flood data."""
    # Start with a clean map
    map_center = [53.5, -1.2]
    if route_data:
        first_route_coords = next(iter(route_data.values()))['coords']
        if first_route_coords:
            map_center = [first_route_coords[0][1], first_route_coords[0][0]] # Lat, Lon for Folium

    hazard_map = folium.Map(location=map_center, zoom_start=8)

    # Add flood overlays to map
    if all_flood_rasters:
        st.write(f"Adding {len(all_flood_rasters)} flood overlays to the map...")
        for flood_data in all_flood_rasters:
            folium.raster_layers.ImageOverlay(
                image=flood_data['overlay_image'],
                bounds=flood_data['bounds'],
                opacity=0.6,
                name="Floodwater Overlay"
            ).add_to(hazard_map)

    # Add routes and hazard markers
    colors = ["red", "blue", "green", "purple", "orange"]
    for vehicle_id, route_info in route_data.items():
        route_coords = route_info["coords"]
        vehicle_num = int(vehicle_id.split('_')[1])
        color = colors[vehicle_num % len(colors)]

        try:
            st.write(f"Drawing route for Vehicle {vehicle_num + 1}...")
            directions = get_directions(tuple(route_coords))
            folium.GeoJson(
                directions,
                style_function=lambda x, c=color: {'color': c, 'weight': 4, 'opacity': 0.8},
                tooltip=f"Vehicle {vehicle_num + 1} - Original Route"
            ).add_to(hazard_map)
        except Exception as e:
            st.warning(f"Could not draw route for vehicle {vehicle_num + 1}: {e}")

        affected_segments = analyze_road_flood_impact(route_coords, all_flood_rasters)
        if affected_segments:
            st.error(f"🚨 Vehicle {vehicle_num + 1} route affected by flooding!")
            for segment in affected_segments:
                folium.Marker(
                    location=[segment["coordinate"][1], segment["coordinate"][0]],
                    popup=f"⚠️ HAZARD DETECTED on Route {vehicle_num + 1}",
                    icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
                ).add_to(hazard_map)
        else:
            st.success(f"✅ Vehicle {vehicle_num + 1} route clear")

    folium.LayerControl().add_to(hazard_map)
    return hazard_map


def render_hazard_analysis_tab():
    """Render the route hazard analysis tab."""
    st.markdown(
        "Monitor real-time hazards and get alternative routing recommendations.")

    # Check if route data exists
    if "route_data" not in st.session_state or not st.session_state.route_data:
        st.info(
            "👆 Please generate a route with at least one destination in the Route Planner tab first.")
        return

    # Initialize session state
    if 'run_hazard_analysis' not in st.session_state:
        st.session_state.run_hazard_analysis = False

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("⚙️ Analysis Settings")

        # Date input
        analysis_date = st.date_input(
            "Analysis date:",
            value=datetime.date(2019, 11, 14),
            help="Select the date for hazard analysis"
        )

        # Tile configuration
        tile_size = render_tile_configuration()

        # Calculate bounding box for current routes
        all_coords = [coord for route in st.session_state.route_data.values()
                      for coord in route['coords']]

        if all_coords:
            current_bbox = (
                min(c[0] for c in all_coords) - 0.01,
                min(c[1] for c in all_coords) - 0.01,
                max(c[0] for c in all_coords) + 0.01,
                max(c[1] for c in all_coords) + 0.01
            )

            # Show tile metrics
            estimated_tiles = render_tile_metrics(current_bbox, tile_size)

            # Action buttons
            if st.button(f"🔍 Check Route Hazards ({len(estimated_tiles)} tiles)", type="primary"):
                st.session_state.run_hazard_analysis = True
                st.session_state.selected_tile_size = tile_size
                st.rerun()

            if st.button("🗑️ Clear Analysis"):
                for key in ["analysis_results", "run_hazard_analysis"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Display results summary
        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results
            total_affected = results.get("total_affected_vehicles", 0)
            flood_data_found = results.get("flood_data_found", False)

            if total_affected > 0:
                st.error(f"⚠️ {total_affected} vehicle route(s) affected by hazards!")
                st.subheader("🚨 Immediate Actions Required")
                st.markdown("- 🛑 **STOP** affected vehicles\n- 📞 Contact drivers\n- 🗺️ Generate alternative routes")
                if st.button("🔄 Generate Alternative Routes"):
                    st.success("✅ Alternative routes generated!")
            else:
                if flood_data_found:
                    st.success("✅ All routes clear - no hazards detected.")
                else:
                    st.warning("⚠️ Could not retrieve flood data. Proceed with caution.")

    with col1:
        st.subheader("🗺️ Route Hazard Analysis Map")

        # Run analysis when button is clicked
        if st.session_state.get('run_hazard_analysis'):
            selected_tile_size = st.session_state.get('selected_tile_size', 0.1)
            
            total_affected, flood_data_found, all_rasters = perform_hazard_analysis(
                st.session_state.route_data, selected_tile_size, analysis_date
            )

            # Store only the data results in session state
            st.session_state.analysis_results = {
                "total_affected_vehicles": total_affected,
                "flood_data_found": flood_data_found,
                "all_flood_rasters": all_rasters
            }
            
            st.session_state.run_hazard_analysis = False
            st.rerun()

        # If there are results, create and display the map
        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results
            hazard_map = create_hazard_map(
                st.session_state.route_data, 
                results.get("all_flood_rasters", [])
            )
            st_folium(hazard_map, width=800, height=600, key="hazard_map_final")
        else:
            st.info("Click 'Check Route Hazards' on the right to view the analysis map.")


def render_geotiff_analysis_tab():
    """Render the GeoTIFF analysis tab."""
    st.markdown(
        "Upload a GeoTIFF file to analyze flood impact on road networks.")

    uploaded_file = st.file_uploader(
        "Upload a GeoTIFF file (.tif, .tiff)",
        type=["tif", "tiff"],
        key="disaster_upload"
    )

    if uploaded_file is not None:
        with st.spinner('Processing GeoTIFF and analyzing road impact...'):
            try:
                with rasterio.open(io.BytesIO(uploaded_file.read())) as src:
                    # Reproject to Web Mercator
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
                        resampling=Resampling.nearest
                    )

                    # Get bounds in WGS84
                    dst_bounds = rasterio.transform.array_bounds(
                        height, width, transform)
                    wgs84_bounds = rasterio.warp.transform_bounds(
                        dst_crs, 'EPSG:4326', *dst_bounds)
                    west, south, east, north = wgs84_bounds

                    # Create map
                    map_center_lat = (south + north) / 2
                    map_center_lon = (west + east) / 2
                    m = folium.Map(
                        location=[map_center_lat, map_center_lon], zoom_start=13)

                    # Add flood overlay
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

                    # Get and analyze road data
                    road_data_overpass = get_road_data(
                        south, west, north, east)
                    road_data_geojson = overpass_to_geojson(road_data_overpass)
                    affected_roads, near_flood_roads = analyze_road_impact(
                        road_data_geojson, reprojected_data[0], transform, dst_crs, 'EPSG:4326'
                    )

                    # Style roads based on flood impact
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

                    # Add roads to map
                    if road_data_geojson['features']:
                        folium.GeoJson(
                            road_data_geojson,
                            name='OpenStreetMap Roads',
                            style_function=style_function,
                            tooltip=folium.GeoJsonTooltip(
                                fields=['name', 'highway'],
                                aliases=['Name:', 'Type:']
                            )
                        ).add_to(m)

                    folium.LayerControl().add_to(m)

                st.success("Analysis complete!")

                # Display results
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


def render():
    """Main render function for disaster management tab."""
    st.header("🚨 Disaster Management & Route Safety")

    disaster_tab1, disaster_tab2 = st.tabs(
        ["Route Hazard Analysis", "GeoTIFF Analysis"])

    with disaster_tab1:
        render_hazard_analysis_tab()

    with disaster_tab2:
        render_geotiff_analysis_tab()
