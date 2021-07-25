import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import io


# filepath = "photo-image-2.png"
# img = Image.open(filepath)
# st.image(img)


def pred_and_plot(filepath, model, class_names):
    img = Image.open(io.BytesIO(filepath))
    img = img.convert('RGB')
    img = img.resize((512, 512), Image.NEAREST)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.

    # Making predictions
    prediction = model.predict(tf.expand_dims(img, axis=0))

    pred_class = class_names[int(tf.round(prediction[0, 0]).numpy())]

    # plot the predictions
    origimg = Image.open(io.BytesIO(filepath))
    col1, col2, col3 = st.beta_columns([1,6,1])
    with col2:
        st.header("Prediction:\t" + pred_class)
        st.image(origimg, width=500)


model = tf.keras.models.load_model("mask_detection_model")

st.title("Face Mask Detection")
filepath = st.file_uploader("Upload an image")
class_names = np.array(['Mask On', 'No Mask'])
print("\n\n\n\n",filepath,"\n\n\n\n")
if filepath:
    print("\n\n\n\n",filepath,"\n\n\n\n")
    pred_and_plot(filepath.read(), model, class_names)
else:
    st.info("Upload an image")




    # img = tf.io.read_file(filepath)
    # img = tf.convert_to_tensor(np.array(img))
    # print("\n\n\n\n",img.shape,"\n\n\n\n")
    # img = tf.io.read_file(img)
    # img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    # img = tf.image.resize(img, (512, 512))