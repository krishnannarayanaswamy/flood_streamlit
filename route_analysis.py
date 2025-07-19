import streamlit as st
import openrouteservice
import numpy as np
import requests
import json

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


def get_flood_overlay_from_langflow(bbox, analysis_date):
    """
    Sends route coordinates and a single date to the Langflow endpoint.
    Returns structured data for flood detection.
    """
    url = "http://localhost:7860/api/v1/run/c496e528-0a6d-4be4-a4a7-f569309e1914"
    api_key = st.secrets.get("LANGFLOW_API_KEY", "")

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    # Create the data structure as expected by your Langflow component
    input_data = {
        "bounding_box": ",".join(map(str, bbox)),
        "analysis_date": analysis_date
    }

    # The input to your flow should be a JSON string of this data
    input_value_string = json.dumps(input_data)

    payload = {
        "input_value": input_value_string,
        "output_type": "chat",
        "input_type": "chat"
    }

    try:
        st.write(f"🔍 Calling Langflow for tile: {bbox}")

        response = requests.post(url, headers=headers,
                                 json=payload, timeout=60)
        response.raise_for_status()

        st.write("✅ Received response from Langflow")

        response_data = response.json()
        if 'outputs' in response_data and response_data['outputs']:
            # Navigate through the nested structure to get the final message text
            message_text = response_data['outputs'][0]['outputs'][0]['results']['message']['text']

            # Log what we received
            st.write(f"📝 Response text: {message_text[:200]}...")

            # The response should contain either a URL or JSON data
            # Return the raw text for now - the flood_detection module will parse it
            return message_text
        else:
            st.warning(
                "Langflow response did not contain the expected output format.")
            return None

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
