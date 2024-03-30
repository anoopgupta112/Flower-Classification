import streamlit as st  
import tensorflow as tf  
import numpy as np  
from PIL import Image 

# Function to load the pre-trained model and cache it for optimized performance
@st.cache_resource(experimental_allow_widgets=True)
def load_model():
    model = tf.keras.models.load_model('flower_model_trained.hdf5')  
    return model 

# Function to predict the class of the input image using the loaded model
def predict_class(image, model):
    image = tf.cast(image, tf.float32)  # image data to float32 datatype
    image = tf.image.resize(image, [180, 180])  # Resizing the input image to match the model's input shape
    image = np.expand_dims(image, axis=0)  # Adding an extra dimension to match the model's input requirements
    prediction = model.predict(image)  # Making predictions using the model
    return prediction  # Returning the predicted class probabilities



def main():
    st.title('Flower Classifier') 

    file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])

    if file is None:
        st.text('Waiting for upload....')
        # Displaying a message if no file is uploaded yet

    else:
        slot = st.empty()
        slot.text('Running inference....')
        test_image = Image.open(file)
        st.image(test_image, caption="Input Image", width=400)
        pred = predict_class(np.asarray(test_image), model)
        class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        result = class_names[np.argmax(pred)]
        # Determining the predicted class by selecting the one with the highest probability

        output = 'The image is a ' + result
        slot.text('Done')
        st.success(output)

if __name__ == "__main__":
    main()

