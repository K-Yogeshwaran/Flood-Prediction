import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Sidebar navigation
menu = ["Home", "Blog", "About"]
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", menu)

if choice == "Home":
    st.title("Flood Prediction System")

    st.write("Enter the following details to predict flood risk:")

    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)
    river_level = st.number_input("River Water Level (meters)", min_value=0.0, max_value=20.0, step=0.1)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)

    if st.button("Predict Flood Risk"):
        input_features = np.array([[rainfall, river_level, soil_moisture, temperature]])
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Flood Risk detected! (Probability: {probability:.2f})")
        else:
            st.success(f"No significant flood risk. (Probability: {probability:.2f})")

elif choice == "Blog":
    st.title("Flood Awareness Blog")

    # Safety tips section
    st.subheader("🛟 What to Do During a Flood")
    st.markdown("""
**Before a Flood:**
- Know your area's flood evacuation routes.
- Prepare an emergency kit with food, water, flashlight, batteries, and first-aid.
- Keep important documents in waterproof containers.
- Monitor weather alerts and warnings from reliable sources.

**During a Flood:**
- Evacuate immediately if advised by authorities.
- Move to higher ground and stay away from rivers, streams, and drains.
- Avoid walking or driving through floodwaters. Just 6 inches of moving water can knock you down.
- Keep updated via local radio, TV, or mobile apps.
- Disconnect electrical appliances to prevent shock.

**If Trapped:**
- Move to the highest level of your home. Only go to the roof if necessary.
- Call emergency services and signal for help.

**After a Flood:**
- Return home only when authorities declare it safe.
- Avoid contact with floodwater – it may be contaminated.
- Disinfect all surfaces and items that came into contact with floodwaters.
- Beware of mold and structural damage in buildings.
    """)

    # Blog post section
    if "blogs" not in st.session_state:
        st.session_state.blogs = []

    st.subheader("📝 Share Your Experience or Tips")
    new_blog = st.text_area("Write your thoughts...")
    if st.button("Submit Blog Post"):
        if new_blog.strip():
            st.session_state.blogs.insert(0, {"username": "user123", "content": new_blog})
            st.success("Blog post submitted!")
        else:
            st.warning("Please write something before submitting.")

    st.subheader("📚 Previous Posts:")
    for blog in st.session_state.blogs:
        st.markdown(f"**@{blog['username']}**")
        st.write(blog["content"])
        st.markdown("---")

elif choice == "About":
    st.title("About Flood Prediction")
    st.write("""
    This app predicts the likelihood of flooding based on environmental features like rainfall, river water level,
    soil moisture, and temperature using a simple logistic regression machine learning model.
    """)
