import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="AI for Agriculture",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 AI for Agriculture")
st.write("Upload a crop leaf image to get an experimental AI prediction.")

# Load both trained models
@st.cache_resource
def load_models():
    tomato_model = tf.keras.models.load_model(
        "tomato_disease_model.keras"
    )
    potato_model = tf.keras.models.load_model(
        "potato_disease_model.keras"
    )
    return tomato_model, potato_model

try:
    tomato_model, potato_model = load_models()
except Exception as e:
    st.error("Model loading failed. Check that both model files exist.")
    st.code(str(e))
    st.stop()

# Class names must match the order used during training
tomato_class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus"
]

potato_class_names = [
    "Early Blight",
    "Healthy",
    "Late Blight"
]

# Basic crop-care suggestions
recommendations = {
    "Bacterial Spot": "Avoid wetting leaves and seek local agricultural advice.",
    "Early Blight": "Remove affected leaves and avoid watering foliage.",
    "Healthy": "Continue regular crop monitoring and suitable watering.",
    "Late Blight": "Separate affected plants and seek agricultural expert advice.",
    "Leaf Mold": "Improve airflow and avoid excess moisture around leaves.",
    "Septoria Leaf Spot": "Remove affected leaves and keep the area clean.",
    "Spider Mites": "Check the undersides of leaves for mites and webbing.",
    "Target Spot": "Remove badly affected leaves and monitor the crop.",
    "Tomato Mosaic Virus": "Avoid handling healthy plants after affected plants; seek expert advice.",
    "Tomato Yellow Leaf Curl Virus": "Check for whiteflies and consult an agricultural expert."
}

# Choose crop
crop = st.selectbox(
    "🌱 Select Crop",
    ["Tomato", "Potato"]
)

# Upload image
uploaded_image = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    try:
        # Read and display image
        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption=f"Uploaded {crop} Leaf",
            use_container_width=True
        )

        # Prepare image
        if crop == "Tomato":
            image = image.resize((128, 128))
        else:
            image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # Select correct model and class names
        if crop == "Tomato":
            selected_model = tomato_model
            selected_class_names = tomato_class_names
        else:
            selected_model = potato_model
            selected_class_names = potato_class_names

        # Make prediction
        if st.button("🔍 Predict Disease"):
            prediction = selected_model.predict(
                image_array,
                verbose=0
            )

            predicted_index = int(np.argmax(prediction[0]))
            confidence = float(
                np.max(prediction[0]) * 100
            )

            if predicted_index >= len(selected_class_names):
                st.error("Model classes do not match the class list.")
            else:
                disease = selected_class_names[predicted_index]

                st.subheader("AI Prediction")
                st.write(f"**Predicted class:** {disease}")
                st.write(f"**Model score:** {confidence:.2f}%")

                st.warning(
                    "This is an experimental student project. "
                    "The score is not a verified probability of disease. "
                    "Please confirm with an agricultural expert."
                )

                st.subheader("💡 Crop Care Recommendation")
                st.write(
                    recommendations.get(
                        disease,
                        "Consult a local agricultural expert for advice."
                    )
                )

    except Exception as e:
        st.error("Could not process the image or make a prediction.")
        st.code(str(e))
else:
    st.info("Please upload a leaf image to continue.")


    
