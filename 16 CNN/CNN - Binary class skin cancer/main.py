import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('skin_cancer_cnn.h5')


# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img


# Streamlit App
st.title("Skin Cancer Detection")

st.markdown("""
    This is a skin cancer detection application. Upload an image, and the model will predict whether the skin lesion is **Malignant** or **Benign**.
""")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Predict and display results
    class_label, img = predict_skin_cancer(uploaded_image, model)

    # Display the result image with the prediction title
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  # Show the uploaded image first
    st.write(f"**Prediction: {class_label}**")

    # # Display processed result image with the prediction title
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # ax.set_title(f"Predicted: {class_label}")
    # ax.axis("off")
    # st.pyplot(fig)

# Additional info and styling
st.markdown("""
    ### About the Model:
    This model uses CNN architecture for predicting whether a skin lesion is **Benign** or **Malignant** based on images of skin lesions.

    #### Features:
    - **Input**: Skin lesion images
    - **Output**: **Benign** or **Malignant** classification

    #### How to use:
    1. Upload an image of a skin lesion.
    2. The model will predict if it's **Benign** or **Malignant**.
""")
