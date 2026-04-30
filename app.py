import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from pathlib import Path
import shutil

# --- Page Config ---
st.set_page_config(page_title="Karachi Real Estate Intelligence", layout="wide")

# --- Custom CSS for the UI Reference ---
st.markdown("""
    <style>
    .result-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .result-value {
        font-size: 48px;
        font-weight: bold;
        margin: 0;
    }
    .actual-val {
        color: #666;
        font-size: 14px;
    }
    .insight-label {
        font-size: 14px;
        color: #888;
        margin-bottom: 5px;
    }
    .insight-value {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model & Data ---
MODEL_PATH = Path("property_model.pkl")
MODEL_PARTS_DIR = Path("property_model.pkl.parts")


def ensure_model_file() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    if not MODEL_PARTS_DIR.exists():
        raise FileNotFoundError(
            "property_model.pkl is missing and no chunk directory was found. "
            "Run split_model.py to create property_model.pkl.parts before shipping the repo."
        )

    chunk_paths = sorted(MODEL_PARTS_DIR.glob("property_model.pkl.part*"))
    if not chunk_paths:
        raise FileNotFoundError(
            "property_model.pkl is missing and the chunk directory is empty."
        )

    temp_path = MODEL_PATH.with_suffix(".pkl.tmp")
    with temp_path.open("wb") as target_file:
        for chunk_path in chunk_paths:
            with chunk_path.open("rb") as source_file:
                shutil.copyfileobj(source_file, target_file)

    temp_path.replace(MODEL_PATH)
    return MODEL_PATH


@st.cache_resource
def load_assets():
    model = joblib.load(ensure_model_file())
    le_type = joblib.load('le_type.pkl')
    le_neigh = joblib.load('le_neigh.pkl')
    df = pd.read_csv('zameen_karachi_cleaned_final.csv')
    return model, le_type, le_neigh, df

model, le_type, le_neigh, df = load_assets()

# --- Geographic Lookup ---
geo_lookup = {
    "DHA Defence": [24.8138, 67.0671], "Clifton": [24.8133, 67.0298], "Bath Island": [24.8322, 67.0305],
    "Civil Lines": [24.8465, 67.0295], "Saddar": [24.8605, 67.0251], "Cantt": [24.8485, 67.0392],
    "Gulshan-e-Iqbal": [24.9157, 67.0931], "Gulistan-e-Jauhar": [24.9107, 67.1261], "Scheme 33": [24.9722, 67.1271],
    "PECHS": [24.8690, 67.0655], "Bahadurabad": [24.8825, 67.0722], "Garden East": [24.8775, 67.0325],
    "Garden West": [24.8732, 67.0225], "Karsaz": [24.8885, 67.0905],
    "North Nazimabad": [24.9372, 67.0422], "Nazimabad": [24.9168, 67.0322], "Gulberg": [24.9350, 67.0750],
    "Federal B Area": [24.9300, 67.0800], "Liaquatabad": [24.9085, 67.0355],
    "Surjani Town": [25.0250, 67.0585], "North Karachi": [24.9750, 67.0650], "New Karachi": [24.9900, 67.0700],
    "Orangi Town": [24.9450, 66.9850], "Baldia Town": [24.9050, 66.9750], "SITE": [24.8950, 67.0150],
    "Malir": [24.8961, 67.1950], "Malir Cantt": [24.9150, 67.2050], "Model Colony": [24.9055, 67.1750],
    "Korangi": [24.8366, 67.1200], "Landhi": [24.8450, 67.1550], "Shah Faisal Town": [24.8850, 67.1450],
    "Bin Qasim": [24.8150, 67.3350], "Bahria Town Karachi": [24.9961, 67.3181],
    "Gadap Town": [25.1000, 67.2000], "Super Highway": [24.9850, 67.1550]
}

# --- State Management ---
if 'selected_neigh' not in st.session_state:
    st.session_state.selected_neigh = "DHA Defence"
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None
if "pred_area" not in st.session_state:
    st.session_state.pred_area = 120
if "last_map_popup" not in st.session_state:
    st.session_state.last_map_popup = None

# --- Main App ---
st.title("🏙️ Karachi Real Estate Intelligence Dashboard")

tabs = st.tabs(["🎯 Location & Prediction", "📊 Market Analysis", "💰 Investment Insight"])

with tabs[0]:
    col_input, col_display = st.columns([1, 2])

    with col_input:
        st.subheader("🛠️ Property Specs")
        with st.form("pred_form"):
            sorted_neigh = sorted(le_neigh.classes_)
            default_idx = sorted_neigh.index(st.session_state.selected_neigh) if st.session_state.selected_neigh in sorted_neigh else 0

            neigh = st.selectbox("Select Neighborhood", sorted_neigh, index=default_idx)
            p_type = st.selectbox("Property Type", le_type.classes_)
            area = st.number_input("Area (Sq. Yards)", value=120)

            c_bed, c_bath = st.columns(2)
            beds = c_bed.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8], index=2)
            baths = c_bath.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 7, 8], index=2)

            st.write("**Premium Features**")
            w_open = st.checkbox("West Open")
            park = st.checkbox("Parking Space")
            furn = st.checkbox("Furnished")
            inst = st.checkbox("Installment Plan Available")

            submit = st.form_submit_button("Calculate Market Value", use_container_width=True)

    if submit:
        st.session_state.selected_neigh = neigh
        t_enc = le_type.transform([p_type])[0]
        n_enc = le_neigh.transform([neigh])[0]
        inputs = np.array([[area, beds, baths, int(w_open), int(park), int(furn), int(inst), t_enc, n_enc]])
        val = model.predict(inputs)[0]

        # Update persistent state
        st.session_state.pred_result = val
        st.session_state.pred_area = area

    with col_display:
        if st.session_state.pred_result is not None:
            val = st.session_state.pred_result
            area_used = st.session_state.pred_area

            # 1. Market Valuation Result (White Card)
            st.write("📊 **Market Valuation Result**")
            st.markdown(f"""
                <div class="result-card">
                    <p class="result-value">PKR {val/10000000:.2f} Crore</p>
                    <p class="actual-val">Estimated Value (Actual: PKR {int(val):,})</p>
                </div>
                """, unsafe_allow_html=True)

            # 2. Market Insights
            st.write("🔍 **Market Insights**")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown('<p class="insight-label">Price per SqYd</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="insight-value">{int(val/max(area_used, 1)):,}</p>', unsafe_allow_html=True)
            with m2:
                st.markdown('<p class="insight-label">Market Tier</p>', unsafe_allow_html=True)
                st.markdown('<p class="insight-value">Standard 🏠</p>', unsafe_allow_html=True)
            with m3:
                st.markdown('<p class="insight-label">Area Demand</p>', unsafe_allow_html=True)
                st.markdown('<p class="insight-value">Stable 📈</p>', unsafe_allow_html=True)

            # 3. Area Trend Chart
            st.write("📈 **Area Trend Estimation**")
            trend_df = pd.DataFrame({
                'Year': ['2023', '2024', '2025 (Est)', 'Current'],
                'Price': [val * 0.88, val * 0.94, val * 1.05, val]
            })
            fig = px.line(trend_df, x='Year', y='Price', markers=True, template="plotly_dark")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        # 4. Interactive Map
        st.write("📍 **Location Mapping**")
        # Center map on selected neighborhood if possible, else Karachi center
        map_center = geo_lookup.get(st.session_state.selected_neigh, [24.8607, 67.0011])
        m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB dark_matter")

        for area_name, coords in geo_lookup.items():
            if area_name in le_neigh.classes_:
                is_selected = (area_name == st.session_state.selected_neigh)
                folium.CircleMarker(
                    location=coords,
                    radius=8 if is_selected else 5,
                    color="red" if is_selected else "#3fa7ff",
                    fill=True,
                    fill_opacity=0.9 if is_selected else 0.6,
                    popup=area_name
                ).add_to(m)

        map_data = st_folium(m, width="100%", height=350, key="pred_map")

        # Handle Map Interactions
        clicked = map_data.get("last_object_clicked_popup")
        if clicked:
            clicked = clicked.strip()
            if clicked != st.session_state.last_map_popup:
                st.session_state.last_map_popup = clicked
                if clicked in le_neigh.classes_ and clicked != st.session_state.selected_neigh:
                    st.session_state.selected_neigh = clicked
                    st.rerun()

# --- Tab 2 and Tab 3 remain unchanged ---
with tabs[1]:
    st.subheader("Regional Comparison")
    available_options = list(le_neigh.classes_)
    valid_defaults = [d for d in [st.session_state.selected_neigh, "Gulshan-e-Iqbal"] if d in available_options]
    areas = st.multiselect("Select Areas to Compare", options=available_options, default=valid_defaults)
    if areas:
        filtered_df = df[df['Neighborhood'].isin(areas)]
        fig = px.box(filtered_df, x="Neighborhood", y="Price_Cleaned", color="Neighborhood")
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Opportunity Finder")
    search = st.text_input("Search keywords (e.g., 'corner', 'emergency')")
    if search:
        hits = df[df['Description'].str.contains(search, case=False, na=False)].head(10)
        for _, row in hits.iterrows():
            with st.expander(f"{row['Title']} - {row['Price_Cleaned']/10000000:.2f} Cr"):
                st.write(row['Description'])