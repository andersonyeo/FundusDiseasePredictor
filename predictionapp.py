import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam

# Sidebar with app information
st.sidebar.title("About")
st.sidebar.info(
    """
    **Fundus Disease Prediction App** \n
    Upload a fundus image, and the model will predict the disease. \n
    Supported diseases: CNV, DME, DRUSEN, NORMAL.
    
    CNV: Choroidal Neovascularization
    DME: Diabetic Macular Edema
    DRUSEN (yellow deposits under the retina)

    **Disclaimer:** This app is for informational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """
)

# Define model architecture function
def create_model():
    vgg16_model = VGG16(include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(vgg16_model)
    for layer in model.layers:
        layer.trainable = False
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the weights
weights_path = "model_vgg16_trained(2349spe).weights.h5" 
try:
    model.load_weights(weights_path)
except Exception as e:
    st.error(f"Error loading weights: {e}")

# Main app interface
st.title('Fundus Disease Prediction App')
st.write("Upload a fundus image, and the model will predict the disease.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    # Display the image in the first column
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Resize image to match model's expected input
    img_array = np.array(img.convert("RGB"))  # Ensure image has 3 color channels
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array / 255.0  # Normalize to [0,1] as done during training

    # Predict diseases
    if st.button('Predict'):
        try:
            prediction = model.predict(img_array)
            # Assuming the model outputs a softmax vector
            classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            predicted_class = classes[np.argmax(prediction)]
            confidence_scores = prediction[0] * 100  # Convert to percentage

            # Display prediction results in the second column
            with col2:
                st.markdown(f'### Prediction: **{predicted_class}**')
                st.markdown("### Confidence Scores:")
                for i, score in enumerate(confidence_scores):
                    st.markdown(f"**{classes[i]}:** {score:.2f}%")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Add the disclaimer at the bottom of the main interface
st.write(
    """
    **Disclaimer:** This app is for informational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """
)

