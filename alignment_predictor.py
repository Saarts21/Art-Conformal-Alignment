import numpy as np
import pandas as pd
import requests, time, re
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from access_token import HF_TOKEN

# Prepare features for g - the alignment predictor
# g(X_i,f(X_i)) = A_hat

# Hugging face models
OBJECT_DETECTION_API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.5

ZERO_SHOT_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
ZERO_SHOT_CONFIDENCE_THRESHOLD = 0.5

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Artist predictor model
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
NUM_OF_ARTISTS = 10
columns = ["name", "paintings", "class_weight"]
artists_top = pd.DataFrame(data, columns=columns)


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
    tries = 3
    while tries > 0:
        try:
            detections = image_query(OBJECT_DETECTION_API_URL, generated_image)
            # filter objects with low confidence
            objects_dict = {d['label']: d['score'] for d in detections if d['score'] > OBJECT_DETECTION_CONFIDENCE_THRESHOLD}
            return list(objects_dict.keys())
        except Exception as e:
            print(e)
            print(detections)
            tries -= 1
            time.sleep(1)
    print("object detection failed")
    return []


def extract_artist_from_prompt(prompt):
    """
    Assuming the beginning of the prompt is in this format:
    'Draw the painting by Vincent Van Gogh as following.'
    """
    words = prompt.split(" ")
    artist = words[4]
    return artist


def artists_prediction_probabilities(generated_image_path):
    test_image = load_img(generated_image_path, target_size=(224, 224))
    test_image = img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    prediction_probabilities = prediction[0]
    return prediction_probabilities


def count_objects_in_prompt(prompt, objects):
    """
    Uses a zero-shot LLM model to examine which objects in the generated image
    are mentioned in the prompt in high confidence.
    Returns the number of predicted objects.
    """
    # no objects found
    if len(objects) == 0:
        return 0
    
    count = 0
    for obj in objects:
        if obj in prompt:
            count += 1
            continue

        tries = 3
        while tries > 0:
            try:
                output = text_query(ZERO_SHOT_API_URL, {"inputs": prompt,
                                                        "parameters": {"candidate_labels": [obj]},
                                                        })
                labels = output['labels']
                scores = output['scores']
                print(scores)

                # filter labels with low confidence
                output_dict = {labels[i]: scores[i] for i in range(len(labels)) if scores[i] > ZERO_SHOT_CONFIDENCE_THRESHOLD}
                count += len(output_dict.keys())
                break
                
            except Exception as e:
                print(e)
                print(output)
                tries -= 1
                print(f"sleeping for 1 sec, tries left: {tries}")
                time.sleep(1)

    return count


def artist_string_to_index(artist_string):
    artist_string = artist_string.replace('_', ' ')
    for index, row in artists_top.iterrows():
        if artist_string == row['name']:
            return index

def feature_extraction(prompt, generated_image):
    """
    generated_image is path
    Returns a numpy vector with real numbers as features
    """
    # compare artist
    artist_string = extract_artist_from_prompt(prompt)
    artist_index = artist_string_to_index(artist_string)
    ground_truth_artist = np.zeros(NUM_OF_ARTISTS)
    ground_truth_artist[artist_index] = 1.0
    predicted_prob_artists = artists_prediction_probabilities(generated_image)

    # compare objects
    generated_image_objects = extract_objects_from_generated_image(generated_image)
    shared_elements_count = count_objects_in_prompt(prompt, generated_image_objects)
    shared_elements_count = float(shared_elements_count)

    return ground_truth_artist, predicted_prob_artists, shared_elements_count

def extract_features_from_data():
    table_path = "data.csv"
    df = pd.read_csv(table_path)

    df['ground_truth_artist_vector'] = pd.NA
    df['predicted_prob_artists_vector'] = pd.NA
    df['shared_elements_count'] = pd.NA

    features_dict = {
        "ground_truth_artist_vector": [],
        "predicted_prob_artists_vector": [],
        "shared_elements_count": []
    }

    for i, row in df.iterrows():
        prompt = row['prompt']
        generated_image_path = row['generated_artwork_name']
        ground_truth_artist, predicted_prob_artists, shared_elements_count = feature_extraction(prompt, generated_image_path)
        
        features_dict["ground_truth_artist_vector"].append(ground_truth_artist)
        features_dict["predicted_prob_artists_vector"].append(predicted_prob_artists)
        features_dict["shared_elements_count"].append(shared_elements_count)

    # save results
    df['ground_truth_artist_vector'] = features_dict['ground_truth_artist_vector']
    df['predicted_prob_artists_vector'] = features_dict['predicted_prob_artists_vector']
    df['shared_elements_count'] = features_dict['shared_elements_count']
    df.to_csv(table_path, index=False)


def parse_out_objects():
    table_path = "data.csv"
    df = pd.read_csv(table_path)
    results = []

    for i, row in df.iterrows():
        if i == 159 or i == 732:
            print(f"--- row {i} ---")

            prompt = row['prompt']
            generated_image_path = row['generated_artwork_name']

            objects_list = extract_objects_from_generated_image(generated_image_path)

            print(f"objects list: {objects_list}")

            shared_elements_count = count_objects_in_prompt(prompt, objects_list)
            shared_elements_count = float(shared_elements_count)
            print(f"shared_elements_count: {shared_elements_count}")

            results.append(shared_elements_count)
            df.at[i, 'shared_elements_count'] = shared_elements_count
            df.to_csv(table_path, index=False)

            with open("backup.txt", 'a') as file:
                file.write(f"{i}: {shared_elements_count}\n")

    print(results)
    # df['shared_elements_count'] = results
    # df.to_csv(table_path, index=False)


parse_out_objects()