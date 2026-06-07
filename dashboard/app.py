import json

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Aerospace RUL Dashboard", layout="wide")

st.title("✈️ Aerospace RUL Prediction Dashboard")
st.write(
    "Interactive dashboard for turbofan engine Remaining Useful Life predictions."
)

model_choice = st.selectbox(
    "Choose Prediction Model",
    [
        "LSTM",
        "GRU",
        "SimpleRNN",
        "LinearRegression",
        "RandomForest",
    ]
)

if "data" not in st.session_state:
    st.session_state.data = None

if st.button("Run Prediction"):

    response = requests.get(
        f"http://localhost:8080/api/rul/predict?model={model_choice}"
    )

    if response.status_code == 200:

        raw_text = response.text

        json_start = raw_text.find("[")
        json_end = raw_text.rfind("]") + 1

        if json_start == -1 or json_end == 0:
            st.error("No JSON returned from the API.")
            st.text(raw_text)

        else:
            json_text = raw_text[json_start:json_end]

            st.session_state.data = pd.DataFrame(
                json.loads(json_text)
            )

    else:
        st.error("Prediction API failed.")

if st.session_state.data is not None:

    data = st.session_state.data

    st.success(
        f"{model_choice} prediction completed."
    )

    st.subheader("Fleet Overview")

    total_engines = len(data)

    high_count = len(
        data[data["priority"] == "HIGH"]
    )

    medium_count = len(
        data[data["priority"] == "MEDIUM"]
    )

    low_count = len(
        data[data["priority"] == "LOW"]
    )

    avg_rul = round(
        data["predicted_rul"].mean(),
        2
    )

    min_rul = round(
        data["predicted_rul"].min(),
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Predictions",
        f"{total_engines:,}"
    )

    col2.metric(
        "High Priority",
        f"{high_count:,}"
    )

    col3.metric(
        "Medium Priority",
        f"{medium_count:,}"
    )

    col4.metric(
        "Lowest RUL",
        f"{min_rul:,.2f}"
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Low Priority",
        f"{low_count:,}"
    )

    col6.metric(
        "Average RUL",
        f"{avg_rul:,.2f}"
    )

    col7.metric(
        "Model Used",
        model_choice
    )

    st.subheader("Filter Results")

    priority_filter = st.selectbox(
        "Filter by Priority",
        ["ALL", "HIGH", "MEDIUM", "LOW"]
    )

    filtered_data = data

    if priority_filter != "ALL":

        filtered_data = data[
            data["priority"] == priority_filter
        ]

    display_data = filtered_data.copy()

    display_data = display_data.rename(
        columns={
            "asset_id": "Engine ID",
            "predicted_rul": "Predicted RUL",
            "maintenance_type": "Maintenance Type",
            "priority": "Priority",
            "recommended_action": "Recommended Action",
            "timestamp": "Timestamp",
        }
    )

    display_data["Predicted RUL"] = (
        display_data["Predicted RUL"].map(
            lambda x: f"{x:,.2f}"
        )
    )

    display_data["Timestamp"] = pd.to_datetime(
        display_data["Timestamp"]
    ).dt.strftime("%Y-%m-%d %I:%M:%S %p")

    st.dataframe(
        display_data,
        use_container_width=True
    )

    st.subheader("Priority Summary")

    priority_counts = (
        filtered_data["priority"]
        .value_counts()
        .reset_index()
    )

    priority_counts.columns = [
        "Priority",
        "Count"
    ]

    st.dataframe(
        priority_counts,
        use_container_width=True
    )

    st.bar_chart(
        priority_counts.set_index("Priority")
    )