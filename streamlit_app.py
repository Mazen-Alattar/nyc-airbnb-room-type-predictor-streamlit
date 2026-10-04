import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


st.set_page_config(
    page_title="NYC Airbnb Room Type Predictor",
    page_icon="NYC",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "https://nyc-airbnb-room-type-predictor-go93.onrender.com"
EXAMPLES = {
    "Manhattan / Midtown": {
        "latitude": 40.7484,
        "longitude": -73.9857,
        "price": 120.0,
        "minimum_nights": 2,
        "number_of_reviews": 84,
        "reviews_per_month": 2.3,
        "calculated_host_listings_count": 1,
        "availability_365": 210,
        "neighbourhood_group": "Manhattan",
        "neighbourhood": "Midtown",
    },
    "Brooklyn / Bedford-Stuyvesant": {
        "latitude": 40.6782,
        "longitude": -73.9442,
        "price": 55.0,
        "minimum_nights": 1,
        "number_of_reviews": 210,
        "reviews_per_month": 4.1,
        "calculated_host_listings_count": 3,
        "availability_365": 300,
        "neighbourhood_group": "Brooklyn",
        "neighbourhood": "Bedford-Stuyvesant",
    },
    "Queens / Flushing": {
        "latitude": 40.7282,
        "longitude": -73.7949,
        "price": 38.0,
        "minimum_nights": 3,
        "number_of_reviews": 12,
        "reviews_per_month": 0.6,
        "calculated_host_listings_count": 1,
        "availability_365": 90,
        "neighbourhood_group": "Queens",
        "neighbourhood": "Flushing",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        :root { --ink: #172126; --muted: #657277; --teal: #168f8a; --gold: #f4b860; --paper: #f5f1e8; }
        .stApp { background: radial-gradient(circle at 85% 6%, #dbeae5 0, transparent 28%), var(--paper); color: var(--ink); }
        .block-container { max-width: 1180px; padding-top: 3.5rem; padding-bottom: 3rem; }
        h1, h2, h3, p, label, .stMarkdown { font-family: 'DM Sans', sans-serif; }
        h1 { letter-spacing: -0.04em; font-weight: 700; }
        .eyebrow { color: var(--teal); font-family: 'Space Mono', monospace; font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; }
        .lede { color: var(--muted); font-size: 1.08rem; line-height: 1.6; max-width: 680px; }
        [data-testid='stSidebar'] { background: #172126; }
        [data-testid='stSidebar'] * { color: #f5f1e8 !important; }
        [data-testid='stSidebar'] .stButton button { background: #f4b860; color: #172126 !important; border: 0; }
        div[data-testid='stForm'] { background: rgba(255,255,255,.58); border: 1px solid rgba(23,33,38,.11); border-radius: 12px; padding: 1.2rem 1.4rem .6rem; }
        .result-card { background: #172126; color: #f5f1e8; border-radius: 12px; padding: 1.5rem; min-height: 100%; }
        .result-kicker { color: #8fd5c9; font-family: 'Space Mono', monospace; font-size: .7rem; letter-spacing: .13em; text-transform: uppercase; }
        .result-name { color: #f4b860; font-size: 2rem; font-weight: 700; margin: .35rem 0 1.2rem; }
        .footer-note { color: var(--muted); font-family: 'Space Mono', monospace; font-size: .72rem; margin-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def predict_from_api(api_url: str, values: dict) -> dict:
    request = Request(
        f"{api_url.rstrip('/')}/predict",
        data=json.dumps(values).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(f"Backend rejected the request: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the FastAPI backend: {error.reason}") from error


def show_prediction(result: dict) -> None:
    prediction = result["Predicted_room_type"]
    probabilities = result["Probability"]
    classes = ["Entire home/apt", "Private room", "Shared room"]
    probability_map = dict(zip(classes, probabilities))

    st.markdown(
        f"<div class='result-card'><div class='result-kicker'>Most likely room type</div>"
        f"<div class='result-name'>{prediction}</div></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Model confidence")
    for room_type in classes:
        probability = float(probability_map[room_type])
        st.markdown(f"**{room_type}**  `{probability:.1%}`")
        st.progress(probability, text=None)


def main() -> None:
    inject_styles()
    st.markdown("<div class='eyebrow'>NYC Airbnb / Room Type Predictor</div>", unsafe_allow_html=True)
    st.title("What kind of stay is this listing?")
    st.markdown(
        "<p class='lede'>Enter a listing's details and the trained model will estimate whether it is an entire home, a private room, or a shared room.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Quick start")
        api_url = st.text_input("FastAPI backend URL", DEFAULT_API_URL).strip()
        example_name = st.selectbox("Load an example", ["None", *EXAMPLES])
        if st.button("Use selected example", use_container_width=True) and example_name != "None":
            st.session_state["example"] = EXAMPLES[example_name]
            st.rerun()
        st.caption("The model uses the listing location, price, stay length, availability, reviews, and host activity.")

    defaults = st.session_state.get("example", {})
    with st.form("listing_form"):
        st.subheader("Listing details")
        location, stay, reviews = st.columns(3)
        with location:
            st.markdown("**Location**")
            latitude = st.number_input("Latitude", -90.0, 90.0, float(defaults.get("latitude", 40.7128)), format="%.6f")
            longitude = st.number_input("Longitude", -180.0, 180.0, float(defaults.get("longitude", -74.0060)), format="%.6f")
            neighbourhood_group = st.selectbox("Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], index=["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"].index(defaults.get("neighbourhood_group", "Manhattan")))
            neighbourhood = st.text_input("Neighbourhood", defaults.get("neighbourhood", "Midtown"))
        with stay:
            st.markdown("**Pricing and stay**")
            price = st.number_input("Price per night (USD)", min_value=1.0, value=float(defaults.get("price", 150.0)), step=1.0)
            minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=int(defaults.get("minimum_nights", 3)), step=1)
            availability_365 = st.slider("Days available per year", 0, 365, int(defaults.get("availability_365", 180)))
        with reviews:
            st.markdown("**Reviews and host**")
            number_of_reviews = st.number_input("Total reviews", min_value=0, value=int(defaults.get("number_of_reviews", 24)), step=1)
            reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=float(defaults.get("reviews_per_month", 1.2)), step=0.1)
            calculated_host_listings_count = st.number_input("Listings by this host", min_value=0, value=int(defaults.get("calculated_host_listings_count", 1)), step=1)
        submitted = st.form_submit_button("Predict room type", type="primary", use_container_width=True)

    if submitted:
        neighbourhood_value = (neighbourhood or "").strip()
        if not neighbourhood_value:
            st.error("Enter a neighbourhood before predicting.")
        else:
            values = {
                "latitude": latitude,
                "longitude": longitude,
                "price": price,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": calculated_host_listings_count,
                "availability_365": availability_365,
                "neighbourhood_group": neighbourhood_group,
                "neighbourhood": neighbourhood_value,
            }
            left, right = st.columns([1, 1.2])
            with left:
                try:
                    result = predict_from_api(api_url, values)
                    show_prediction(result)
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    st.error(str(error))
            with right:
                st.subheader("How to read this")
                st.write("The bars show the model's probability for each room type. The prediction is the class with the highest probability.")

    st.markdown("<div class='footer-note'>Trained on NYC Airbnb listing data · Streamlit deployment</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
