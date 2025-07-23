import streamlit as st
import pandas as pd
import requests


def render():
    """Render the driver dashboard tab."""
    st.header("👨‍💼 Driver Dashboard")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Routes", "3")
    col2.metric("Deliveries Today", "47")
    col3.metric("Fuel Efficiency", "8.2 L/100km")

    # Assignments table
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

    # Support agent chat
    st.subheader("🤖 Support Agent")
    st.markdown(
        "Ask questions about weather, traffic conditions, or general FAQs")

    FLOW_URL = "http://localhost:7860/api/v1/run/88df7948-8e1d-4db4-b97e-ce2626d174c3"

    def run_flow(message, output_type="chat", input_type="chat", tweaks=None):
        payload = {"input_value": message,
                   "output_type": output_type, "input_type": input_type}
        if tweaks:
            payload["tweaks"] = tweaks
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(FLOW_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error making API request: {e}")
            return None
        except ValueError as e:
            st.error(f"Error parsing response: {e}")
            return None

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
            with st.spinner("Thinking..."):
                response = run_flow(
                    message=prompt, output_type="chat", input_type="chat")
                if response:
                    try:
                        result = response['outputs'][0]['outputs'][0]['results']['message']['text']
                    except (KeyError, IndexError):
                        result = "I apologize, but I couldn't process your request. Please try again."
                    message_placeholder.markdown(result)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result})
                else:
                    error_message = "I'm having trouble connecting right now. Please try again later."
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message})

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
