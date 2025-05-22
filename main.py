import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model once and cache it for better performance


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")


model = load_model()

# Tensorflow Model Prediction


def model_prediction(test_image):
    try:
        img = Image.open(test_image).convert("RGB").resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        # Convert single image to batch
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# Class labels (defined once)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    # 🌿 Welcome to Team 11's Plant Disease Recognition System! 🔍

    We’re **Team 11**, and we’re excited to assist you in identifying plant diseases quickly and accurately.

    Our goal is to help farmers, gardeners, and agricultural specialists detect plant diseases early—so we can all work together to ensure healthier crops and stronger yields. Simply upload a photo of a plant, and let our intelligent system do the analysis for you.

    ---

    ### 🚀 How It Works

    1. **Upload an Image:** Head to the **Disease Recognition** page and upload a photo of the plant you suspect may be affected.
    2. **Image Analysis:** Our system processes the image using advanced machine learning models to detect signs of disease.
    3. **Get Results:** View the diagnostic results along with helpful recommendations for next steps.

    ---

    ### 🌟 Why Use Team 11's System?

    - **High Accuracy:** Powered by cutting-edge AI techniques to ensure reliable predictions.
    - **Easy to Use:** Clean, user-friendly interface—no technical skills needed.
    - **Fast Results:** Get insights in just seconds to act swiftly.

    ---

    ### 🌱 Get Started

    Click on the **Disease Recognition** page in the sidebar to upload an image and explore the power of Team 11’s Plant Disease Recognition System!

    ---

    ### 👨‍💻 About Team 11

    Want to know more about the creators? Visit the **About** page to learn more about Team 11 and our mission.

    ---
    """)

# About Project
elif app_mode == "About":
    st.header("About Creator")
    st.markdown("## 👥 Team Members - Tanta University")
    members = [
        {"name": "Seif El-Deen Abd El-Fatah", "id": "01091070852"},
        {"name": "Zyad Wael Attia", "id": "01017826329"},
        {"name": "Ahemd Yousef", "id": "01017053481"},
        {"name": "Ahmed Abou-Setah", "id": "01050231883"},
        {"name": "Samir Mohamed Samir", "id": "01016718846"},
        {"name": "Mohamed Maher", "id": "01156656502"},
        {"name": "Seif El-Eslam Mohamed", "id": "01093163091"},
    ]

    for member in members:
        st.markdown(f"""
        ---
        #### {member["name"]}
        Tanta University  
        #### {member["id"]}
        """)

    st.markdown("---")
    st.markdown("### Supervised by Dr/ Hany El-Gaish && Eng/ Mohamed El-Sharkawy ")
    st.markdown(
        "### Special Thanks to Dr/ Hany and Eng/ Mohamed for his support and guidance throughout this project.")
    st.markdown("### We hope you find our system helpful and informative!")

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader(
        "Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                result_index = model_prediction(test_image)
                if result_index is not None:
                    st.success(
                        f"Model Prediction: **{class_names[result_index]}**")
                else:
                    st.error("Prediction failed. Please try another image.")
