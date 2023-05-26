import keras.layers
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D
from keras.utils import img_to_array
from keras.applications.resnet import preprocess_input
import cv2

import numpy as np
import requests
from io import BytesIO
from PIL import Image


def load_and_preprocess_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def create_visual_content_encoder(input_shape=(224, 224, 3)):
    # load pre-trained ReseNet50 without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # flatten the output 3D matrix into a 1D vector
    res_output = base_model.output
    pool_output = GlobalAveragePooling2D()(res_output)
    # x = keras.layers.Flatten()(x)

    # create the VCE model
    model = Model(inputs=base_model.input, outputs=pool_output, name='visual_content_encoder')
    return model


# print(tf.config.list_physical_devices('GPU'))

# # download an image and test the resnet50
# input_shape = (224, 224, 3)
# vce = create_visual_content_encoder(input_shape)
#
# img_url = "https://blog.rtwilson.com/wp-content/uploads/2011/10/len_std.jpg"
# img_array = load_and_preprocess_image_from_url(img_url)
#
# predictions = vce(img_array)
# print(predictions)
