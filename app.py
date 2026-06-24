"""
🌾 Crop Recommendation System
A beautiful Streamlit web application that uses a Random Forest Classifier
to recommend the best crop based on soil and weather conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────── Page Config ────────────────────────────

st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ─────────────────────────────

st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0f1a;
    --bg-card: rgba(17, 25, 40, 0.75);
    --bg-glass: rgba(255, 255, 255, 0.04);
    --border-glass: rgba(255, 255, 255, 0.08);
    --accent-green: #10b981;
    --accent-emerald: #34d399;
    --accent-teal: #14b8a6;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --gradient-1: linear-gradient(135deg, #10b981, #14b8a6);
    --gradient-2: linear-gradient(135deg, #06b6d4, #8b5cf6);
    --gradient-3: linear-gradient(135deg, #f59e0b, #ef4444);
}

/* ── Global Styling ── */
.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #06b6d4, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.hero-header p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(16, 185, 129, 0.08);
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(16, 185, 129, 0.3);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(20, 184, 166, 0.08));
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeInUp 0.6s ease-out;
}
.result-card .crop-emoji {
    font-size: 4rem;
    margin-bottom: 0.5rem;
    animation: bounceIn 0.8s ease-out;
}
.result-card .crop-name {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.result-card .crop-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
}

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #0a1628 100%) !important;
}
section[data-testid="stSidebar"] .stSlider > div > div {
    color: var(--accent-green) !important;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), transparent);
    border: none;
    margin: 1.5rem 0;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes bounceIn {
    0% { transform: scale(0.3); opacity: 0; }
    50% { transform: scale(1.1); }
    70% { transform: scale(0.9); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ── Feature Tag ── */
.feature-tag {
    display: inline-block;
    background: rgba(16, 185, 129, 0.12);
    color: var(--accent-emerald);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 8px;
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Input summary table ── */
.input-summary {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border-glass);
}
.input-summary th {
    background: rgba(16, 185, 129, 0.1);
    color: var(--accent-emerald);
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.input-summary td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border-glass);
    font-size: 0.92rem;
}
.input-summary tr:last-child td {
    border-bottom: none;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Crop Emoji Map ─────────────────────────

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🖤",
    "lentil": "🍲", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
    "orange": "🍊", "papaya": "🥝", "coconut": "🥥", "cotton": "🧶",
    "jute": "🌿", "coffee": "☕",
}

CROP_TIPS = {
    "rice": "Rice thrives in warm, humid climates with abundant water. Consider paddy field irrigation.",
    "maize": "Maize needs well-drained, fertile soil. Plant in rows for optimal sunlight exposure.",
    "chickpea": "Chickpeas prefer cool, dry conditions. They fix nitrogen, improving soil health.",
    "kidneybeans": "Kidney beans need moderate temperatures and consistent moisture during flowering.",
    "pigeonpeas": "Pigeon peas are drought-tolerant and excellent for intercropping systems.",
    "mothbeans": "Moth beans thrive in arid conditions — perfect for low-rainfall regions.",
    "mungbean": "Mung beans grow fast (60-90 days) and enrich soil with nitrogen fixation.",
    "blackgram": "Black gram prefers warm, humid conditions and well-drained loamy soil.",
    "lentil": "Lentils do well in cool seasons. They're a valuable source of protein.",
    "pomegranate": "Pomegranates are drought-tolerant and prefer hot, dry summers.",
    "banana": "Bananas need rich, well-drained soil with consistent moisture and warmth.",
    "mango": "Mangoes thrive in tropical climates. Young trees need protection from frost.",
    "grapes": "Grapes prefer Mediterranean climates with warm, dry summers.",
    "watermelon": "Watermelons need long, warm growing seasons and sandy loam soil.",
    "muskmelon": "Muskmelons love heat. Ensure good drainage to prevent root rot.",
    "apple": "Apples need cold winters for dormancy. Choose varieties suited to your zone.",
    "orange": "Oranges need subtropical conditions with protection from frost.",
    "papaya": "Papayas grow best in tropical climates with consistent warmth and moisture.",
    "coconut": "Coconut palms thrive in coastal, tropical regions with sandy soil.",
    "cotton": "Cotton needs a long, warm growing season with moderate rainfall.",
    "jute": "Jute requires a warm, humid climate with heavy rainfall during growth.",
    "coffee": "Coffee prefers high-altitude tropical regions with shade and consistent rain.",
}


# ──────────────────────────── Load & Train ───────────────────────────

@st.cache_resource(show_spinner=False)
def load_and_train_model():
    """Load the dataset, train a Random Forest Classifier, and return model + metadata."""
    try:
        df = pd.read_csv("Crop_recommendation.csv")
    except FileNotFoundError:
        return None, None, None, None, "❌ **Dataset not found!** Please place `Crop_recommendation.csv` in the same directory as `app.py`."

    # Encode target labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    # Features & target
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features]
    y = df["label_encoded"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, le, df, accuracy, None


# ──────────────────────────── Main App ───────────────────────────────

def main():
    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🌾 Crop Recommendation System</h1>
        <p>Powered by Machine Learning — enter your soil &amp; weather data to discover the ideal crop for your land.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    with st.spinner("🔬 Training the Random Forest model..."):
        model, label_encoder, df, accuracy, error = load_and_train_model()

    if error:
        st.error(error)
        st.info("📥 You can download the dataset from [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).")
        st.stop()

    # ── Sidebar Inputs ──
    with st.sidebar:
        st.markdown("## 🧪 Input Parameters")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("### 🧬 Soil Nutrients")
        N = st.slider("Nitrogen (N)", 0, 140, 50, help="Ratio of Nitrogen content in soil (kg/ha)")
        P = st.slider("Phosphorus (P)", 5, 145, 55, help="Ratio of Phosphorus content in soil (kg/ha)")
        K = st.slider("Potassium (K)", 5, 205, 45, help="Ratio of Potassium content in soil (kg/ha)")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("### 🌡️ Weather Conditions")
        temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, step=0.5, help="Average temperature in degrees Celsius")
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0, step=0.5, help="Relative humidity in percentage")
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=1.0, help="Annual rainfall in millimeters")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("### ⚗️ Soil Property")
        ph = st.slider("Soil pH", 3.5, 9.5, 6.5, step=0.1, help="pH value of the soil (3.5 = acidic, 9.5 = alkaline)")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        predict_btn = st.button("🌱 Recommend Crop", use_container_width=True, type="primary")

    # ── Dashboard Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{accuracy:.1%}</div>
            <div class="metric-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['label'].nunique()}</div>
            <div class="metric-label">Crop Classes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Training Samples</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">7</div>
            <div class="metric-label">Input Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Input Summary & Prediction ──
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("### 📋 Your Input Summary")
        st.markdown(f"""
        <div class="glass-card">
            <table class="input-summary">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>🧬 Nitrogen (N)</td><td><b>{N}</b> kg/ha</td></tr>
                <tr><td>🧬 Phosphorus (P)</td><td><b>{P}</b> kg/ha</td></tr>
                <tr><td>🧬 Potassium (K)</td><td><b>{K}</b> kg/ha</td></tr>
                <tr><td>🌡️ Temperature</td><td><b>{temperature}</b> °C</td></tr>
                <tr><td>💧 Humidity</td><td><b>{humidity}</b> %</td></tr>
                <tr><td>⚗️ Soil pH</td><td><b>{ph}</b></td></tr>
                <tr><td>🌧️ Rainfall</td><td><b>{rainfall}</b> mm</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("### 🌱 Prediction Result")

        if predict_btn:
            # Prepare input
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction_encoded = model.predict(input_data)[0]
            predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]

            # Confidence (probability)
            probabilities = model.predict_proba(input_data)[0]
            confidence = np.max(probabilities) * 100

            # Get emoji and tip
            emoji = CROP_EMOJI.get(predicted_crop.lower(), "🌱")
            tip = CROP_TIPS.get(predicted_crop.lower(), "This crop suits your soil and weather conditions well.")

            # Store in session state for persistence
            st.session_state["prediction"] = {
                "crop": predicted_crop,
                "emoji": emoji,
                "confidence": confidence,
                "tip": tip,
                "probabilities": probabilities,
            }

        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]
            st.markdown(f"""
            <div class="result-card">
                <div class="crop-emoji">{pred['emoji']}</div>
                <div class="crop-name">{pred['crop'].capitalize()}</div>
                <div class="crop-subtitle">Recommended with <b>{pred['confidence']:.1f}%</b> confidence</div>
            </div>
            """, unsafe_allow_html=True)

            st.success(f"💡 **Tip:** {pred['tip']}")

            # ── Top 3 alternatives ──
            top_indices = np.argsort(pred["probabilities"])[::-1][:3]
            top_crops = label_encoder.inverse_transform(top_indices)
            top_probs = pred["probabilities"][top_indices] * 100

            st.markdown("**Top 3 Recommendations:**")
            for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                bar_emoji = CROP_EMOJI.get(crop.lower(), "🌱")
                medal = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"{medal} **{crop.capitalize()}** {bar_emoji}")
                st.progress(prob / 100)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 3rem;">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">🌍</p>
                <p style="color: #94a3b8; font-size: 1.05rem;">
                    Adjust the parameters in the sidebar<br>and click <b>🌱 Recommend Crop</b> to get started.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Dataset Explorer (Expandable) ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    with st.expander("📊 Explore the Dataset", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📈 Statistics", "🌾 Crop Distribution"])

        with tab1:
            st.dataframe(df.drop(columns=["label_encoded"], errors="ignore"), use_container_width=True, height=400)

        with tab2:
            st.dataframe(df.describe().round(2), use_container_width=True)

        with tab3:
            crop_counts = df["label"].value_counts().reset_index()
            crop_counts.columns = ["Crop", "Count"]
            st.bar_chart(crop_counts.set_index("Crop"), use_container_width=True, height=400)

    # ── Feature Importance ──
    with st.expander("🎯 Feature Importance (What Matters Most?)", expanded=False):
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)
        st.bar_chart(feat_df.set_index("Feature"), use_container_width=True, height=350)

    # ── Footer ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 1rem; color: #64748b; font-size: 0.82rem;">
        Built with ❤️ using <b>Streamlit</b> &amp; <b>Scikit-learn</b> &nbsp;|&nbsp;
        Random Forest Classifier &nbsp;|&nbsp; Crop Recommendation Dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
