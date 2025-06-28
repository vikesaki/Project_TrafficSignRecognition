#type:ignore
import os
import streamlit as st

# Dataset storage
import pandas as pd
import numpy as np

# Vizualization
import matplotlib.pyplot as plt

# Model Loading
from tensorflow import keras
from tensorflow.keras.models import load_model
import json
from PIL import Image

# ML
from keras.utils import register_keras_serializable
from keras.metrics import Metric
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

@register_keras_serializable()
class SparseMacroF1Score(Metric):
    def __init__(self, num_classes=5, name='sparse_macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int32)

        for i in range(self.num_classes):
            y_true_i = tf.equal(y_true, i)
            y_pred_i = tf.equal(y_pred_labels, i)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, y_pred_i), self.dtype))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_i), y_pred_i), self.dtype))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, tf.logical_not(y_pred_i)), self.dtype))

            self.tp.assign(tf.tensor_scatter_nd_add(self.tp, [[i]], [tp]))
            self.fp.assign(tf.tensor_scatter_nd_add(self.fp, [[i]], [fp]))
            self.fn.assign(tf.tensor_scatter_nd_add(self.fn, [[i]], [fn]))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1)

    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def interpretation (prediction, label_map):
    try:
        if isinstance(prediction, (list, tuple, np.ndarray)):
            idx = int(prediction[0])
        else:
            idx = int(prediction)

        return label_map.get(str(idx), "Unknown class")

    except Exception as e:
        print(f"Error interpreting prediction: {e}")
        return "Error"
    
# Preload the model and label
with open("deployment/group_labels.json", "r") as f:
    label_map = json.load(f)

model = load_model("deployment/model_2.keras", custom_objects={
    "SparseMacroF1Score": SparseMacroF1Score
})

def show():
    st.title("🚦 Traffic Sign Prediction")
    st.markdown("Upload a traffic sign image to classify it using a deep learning model.")

    uploaded_file = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Show original image
        st.subheader("Uploaded Image")
        _, middle, _ = st.columns([0.5, 1, 0.5])
        with middle:
            st.image(uploaded_file, caption="Original Image", use_container_width =True)

        # Preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        resized_img = img.resize((128, 128))
        
        img_array = img_to_array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result_label = interpretation(predicted_class, label_map)
        confidence = np.max(prediction)

        # Show prediction
        st.subheader("Prediction Result")
        st.success(f"Prediction: This is a traffic sign categorize as **{result_label}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")