
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="AI for Agriculture",
    page_icon="🌱",
    layout="wide"
)

# ---------- DESIGN ----------
st.title("🌱 AI FOR AGRICULTURE")
st.caption("Smart crop disease screening and farmer support")
st.info(
    "Student project: AI results are experimental. "
    "Confirm suspected diseases with an agricultural expert."
)

# ---------- DISEASE INFORMATION ----------
disease_info = {
    "Bacterial Spot": (
        "Bacterial disease causing dark leaf spots.",
        "Avoid overhead watering and ask an agricultural expert about management."
    ),
    "Early Blight": (
        "Fungal disease that can cause brown leaf spots.",
        "Monitor affected leaves, avoid wetting foliage, and seek local advice."
    ),
    "Healthy": (
        "The model classified this image as healthy.",
        "Continue crop monitoring and suitable watering."
    ),
    "Late Blight": (
        "A potentially serious disease affecting leaves and plants.",
        "Seek prompt local agricultural advice and avoid moving infected plant material."
    ),
    "Leaf Mold": (
        "A disease associated with leaf symptoms and humid conditions.",
        "Improve airflow where possible and avoid excessive leaf moisture."
    ),
    "Septoria Leaf Spot": (
        "A fungal disease associated with spots on leaves.",
        "Keep the growing area clean and ask an expert about disease management."
    ),
    "Spider Mites": (
        "Tiny pests that may cause stippling or webbing on leaves.",
        "Inspect leaf undersides and ask an agricultural expert about suitable control."
    ),
    "Target Spot": (
        "A leaf disease that may cause circular spots.",
        "Monitor affected plants and seek local crop-management advice."
    ),
    "Tomato Mosaic Virus": (
        "A viral disease that can affect tomato growth and leaves.",
        "Avoid handling healthy plants after suspect plants and seek expert advice."
    ),
    "Tomato Yellow Leaf Curl Virus": (
        "A viral disease often associated with whitefly transmission.",
        "Check for whiteflies and consult a local agricultural expert."
    )
}

tomato_classes = [
    "Bacterial Spot", "Early Blight", "Healthy", "Late Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
    "Target Spot", "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus"
]

potato_classes = ["Early Blight", "Healthy", "Late Blight"]

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    tomato = tf.keras.models.load_model("tomato_disease_model.keras")
    potato = tf.keras.models.load_model("potato_disease_model.keras")
    return tomato, potato

try:
    tomato_model, potato_model = load_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error("Could not load the crop models. Check model files and dependencies.")
    st.code(str(e))

# ---------- SESSION HISTORY ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- SIDEBAR ----------
page = st.sidebar.radio(
    "🌿 Navigation",
    [
        "Disease Detection",
        "Disease Guide",
        "Crop Care Tips",
        "Farmer Assistant",
        "Prediction History"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Supported crops: Tomato and Potato")

# ---------- DISEASE DETECTION ----------
if page == "Disease Detection":
    st.header("🔬 Crop Disease Detection")
    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Choose your crop", ["Tomato", "Potato"])
        uploaded = st.file_uploader(
            "Upload a clear leaf photo",
            type=["jpg", "jpeg", "png"]
        )

    with col2:
        st.markdown("### 📸 Photo tips")
        st.write("- Use a clear, well-lit image.")
        st.write("- Keep the leaf in focus.")
        st.write("- Avoid blurry or very dark photos.")
        st.write("- Include the affected part of the leaf.")

    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded leaf", width=320)

            if st.button("🔍 Predict Disease", type="primary"):
                if not models_ready:
                    st.error("Models are not ready. Fix the model loading error first.")
                else:
                    if crop == "Tomato":
                        model = tomato_model
                        class_names = tomato_classes
                        image_size = (128, 128)
                    else:
                        model = potato_model
                        class_names = potato_classes
                        image_size = (224, 224)

                    prepared = image.resize(image_size)
                    array = np.asarray(prepared, dtype=np.float32)
                    array = np.expand_dims(array, axis=0)

                    with st.spinner("Analyzing leaf image..."):
                        output = np.asarray(
                            model.predict(array, verbose=0)
                        )[0].flatten()

                    if len(output) != len(class_names):
                        st.error(
                            "Model output and class list do not match. "
                            "Check the model's training class order."
                        )
                    else:
                        index = int(np.argmax(output))
                        disease = class_names[index]
                        score = float(output[index]) * 100

                        description, advice = disease_info[disease]

                        st.markdown("---")
                        st.subheader("🌱 Prediction Result")
                        st.success(f"Model classification: **{disease}**")
                        st.progress(max(0, min(100, int(score))))
                        st.metric("Model score (not verified probability)", f"{score:.1f}%")
                        st.write(description)
                        st.markdown("### 💡 Suggested next step")
                        st.write(advice)

                        if score < 60:
                            st.warning(
                                "The model score is low. Try a clearer photo "
                                "and get the plant checked by an expert."
                            )

                        st.session_state.history.insert(0, {
                            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "Crop": crop,
                            "Prediction": disease,
                            "Model Score (%)": round(score, 2)
                        })

                        st.caption(
                            "AI screening only; this result does not confirm "
                            "a disease or replace expert diagnosis."
                        )
        except Exception as e:
            st.error("Image processing or prediction failed.")
            st.code(str(e))
    else:
        st.info("Choose a crop and upload a leaf image to begin.")

# ---------- DISEASE GUIDE ----------
elif page == "Disease Guide":
    st.header("📚 Crop Disease Guide")
    selected = st.selectbox("Choose a disease", list(disease_info.keys()))
    description, advice = disease_info[selected]
    st.subheader(selected)
    st.write(description)
    st.markdown("### Suggested care")
    st.write(advice)
    st.caption("Advice is general. Confirm diagnosis before treatment.")

# ---------- CROP CARE ----------
elif page == "Crop Care Tips":
    st.header("🌾 Crop Care Tips")
    st.markdown("""
    ### 💧 Watering
    - Water according to crop needs and soil moisture.
    - Avoid unnecessary waterlogging.

    ### 🌱 Soil health
    - Use well-decomposed compost when suitable.
    - Test soil before making major fertilizer changes.

    ### 🍃 Plant monitoring
    - Inspect both sides of leaves regularly.
    - Remove diseased material appropriately.
    - Clean tools and avoid moving plant material between affected and healthy areas.

    ### 🐛 Pest management
    - Check plants regularly for insects and unusual symptoms.
    - Identify the pest before choosing a treatment.
    - Follow local agricultural guidance and product labels.
    """)

# ---------- FARMER ASSISTANT ----------
elif page == "Farmer Assistant":
    st.header("👨‍🌾 Farmer Assistant")
    question = st.selectbox(
        "What do you need help with?",
        [
            "Select a question",
            "How can I check a diseased leaf?",
            "How can I improve soil health?",
            "How should I water crops?",
            "What should I do about pests?",
            "How can I reduce plant disease spread?"
        ]
    )

    answers = {
        "How can I check a diseased leaf?":
            "Take a clear photo, inspect both sides of the leaf, note how symptoms spread, and consult an agricultural expert.",
        "How can I improve soil health?":
            "Consider a soil test, suitable crop rotation, and well-decomposed organic matter according to local advice.",
        "How should I water crops?":
            "Check soil moisture and crop needs. Avoid overwatering and follow local irrigation recommendations.",
        "What should I do about pests?":
            "Identify the pest first, inspect the whole plant, and ask an agricultural expert about suitable control methods.",
        "How can I reduce plant disease spread?":
            "Clean tools, avoid handling healthy plants after affected plants, and follow local guidance for infected material."
    }

    if question != "Select a question":
        st.info(answers[question])

    st.caption("This is a basic built-in assistant, not a live AI or expert service.")

# ---------- HISTORY ----------
elif page == "Prediction History":
    st.header("📋 Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "⬇️ Download History as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="agriculture_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see it here.")

st.markdown("---")
st.caption("AI for Agriculture | Educational project 🌱")
