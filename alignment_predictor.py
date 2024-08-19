import requests
import numpy as np
from access_token import HF_TOKEN
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare features for g - the alignment predictor
# g(X_i,f(X_i)) = A_hat

NUM_OF_ARTISTS = 10

# Hugging face models
OBJECT_DETECTION_API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.8

ZERO_SHOT_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
ZERO_SHOT_CONFIDENCE_THRESHOLD = 0.7

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def image_query(api_url, filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(api_url, headers=headers, data=data)
    return response.json()

def text_query(api_url, payload):
	response = requests.post(api_url, headers=headers, json=payload)
	return response.json()

def extract_objects_from_generated_image(generated_image):
    """
    Returns a list of objects that seems to appear the generated artwork.
    """
    detections = image_query(OBJECT_DETECTION_API_URL, generated_image)

    # filter objects with low confidence
    objects_dict = {d['label']: d['score'] for d in detections if d['score'] > OBJECT_DETECTION_CONFIDENCE_THRESHOLD}
    return list(objects_dict.keys())

def extract_artist_from_prompt(prompt):
    """
    Assuming the beginning of the prompt is in this format:
    'Draw the painting by Vincent Van Gogh as following.'
    """
    words = prompt.split(".")
    words = words[0].split("by ")
    words = words[1].split(" as")
    artist = words[0]
    return artist

def extract_features_from_prompt(prompt):
    artist = extract_artist_from_prompt(prompt)
    return artist


model = tf.keras.models.load_model('../models/artists_precitor_model_improved.h5')
data = [
    ["Vincent van Gogh", 877, 0.445631],
    ["Edgar Degas", 702, 0.556721],
    ["Pablo Picasso", 439, 0.890246],
    ["Pierre-Auguste Renoir", 336, 1.163149],
    ["Albrecht Dürer", 328, 1.191519],
    ["Paul Gauguin", 311, 1.256650],
    ["Francisco Goya", 291, 1.343018],
    ["Rembrandt", 262, 1.491672],
    ["Alfred Sisley", 259, 1.508951],
    ["Titian", 255, 1.532620]
]

# Defining columns
columns = ["name", "paintings", "class_weight"]
artists_top = pd.DataFrame(data, columns=columns)

batch_size = 16
train_input_shape = (224, 224, 3)

def artists_prediction_probabilities(generated_image_path):
    test_image = image.load_img(generated_image_path, target_size=(train_input_shape[0:2]))
    # Predict artist
    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)
    # Get the artist names
    artists_top_name = artists_top['name'].str.replace(' ', '_').values
    prediction = model.predict(test_image)
    prediction_probabilities = prediction[0]
    # Store probabilities in a dictionary
    return prediction_probabilities

def extract_features_from_generated_image(generated_image):
    objects = extract_objects_from_generated_image(generated_image)
    artists_pred_prob = artists_prediction_probabilities(generated_image)
    return objects, artists_pred_prob

def count_objects_in_prompt(prompt, objects):
    """
    Uses a zero-shot LLM model to examine which objects in the generated image
    are mentioned in the prompt in high confidence.
    Returns the number of predicted objects.
    """
    output = text_query(ZERO_SHOT_API_URL, {"inputs": prompt,
        "parameters": {"candidate_labels": objects},
    })
    labels = output['labels']
    scores = output['scores']

    # filter labels with low confidence
    output_dict = {labels[i]: scores[i] for i in range(len(labels)) if scores[i] > ZERO_SHOT_CONFIDENCE_THRESHOLD}
    return len(output_dict.keys())


def artist_string_to_index(artist_string):
    for index, row in artists_top.iterrows():
        if artist_string == row['name']:
            return index


def feature_extraction(prompt, generated_image):
    """
    Returns a numpy vector with real numbers as features
    """
    # prompt features
    artist_string = extract_features_from_prompt(prompt)
    artist_index = artist_string_to_index(artist_string)
    ground_truth_artist = np.zeros(NUM_OF_ARTISTS)
    ground_truth_artist[artist_index] = 1.0

    # generated image features
    generated_image_objects, predicted_prob_artists = extract_features_from_generated_image(generated_image)

    # compare between prompt and generated image
    shared_elements_count = count_objects_in_prompt(prompt, generated_image_objects)
    shared_elements_count = float(shared_elements_count)

    return np.stack((ground_truth_artist, predicted_prob_artists, shared_elements_count))
