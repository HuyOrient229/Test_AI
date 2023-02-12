import streamlit as st
import numpy as np
from PIL import Image
from skimage import transform
from tensorflow import keras

model_4 = keras.models.load_model('model_4.keras')

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (190, 190, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def predict(image):
    pred = model_4.predict(image)
    num_pred = np.round(pred)
    return int(num_pred[0])

def main():
    st.set_page_config(page_title="Classify Dogs and Cats", page_icon=":dog:", layout="wide")
    st.title("Classify Dogs and Cats")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load(uploaded_file)
        num_pred = predict(image)
        if num_pred == 0:
            label = "cat"
        else:
            label = "dog"

        st.image(image[0], use_column_width=True)
        if label == "dog":
            st.success("🐶  Success!")
            st.write("The image is of a **dog**.")
        elif label == "cat":
            st.success("🐱  Success!")
            st.write("The image is of a **cat**.")
        else:
            st.success(f"The image is of a {label}")


if __name__ == '__main__':
    main()
