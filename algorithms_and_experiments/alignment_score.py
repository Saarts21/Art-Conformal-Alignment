import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from PIL import Image
from scipy.spatial.distance import cdist
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

# Compute the alignment score A_i


FIXED_IMG_SIZE = (100, 100)
FIXED_PALETTE_SIZE = 10

num_of_ref_images = os.listdir("alignment_false_references/")
num_of_ref_images = [file for file in num_of_ref_images if "jpeg" in file]
num_of_ref_images = len(num_of_ref_images)
total_comparisons = 2 * num_of_ref_images + 1  # 10 images, 10 random noise, 1 true reference


def gram_matrix(img):
    img = image_preprocess_2(img)
    if len(img.shape) == 2:
        # black and white sketch: add c = 3
        img = np.stack([img] * 3, axis=-1)
    h, w, c = img.shape
    img = img.reshape(c, h * w)
    gram = np.matmul(img, img.transpose()) / (h * w * c)
    return gram


def image_preprocess_2(img):
    if isinstance(img, np.ndarray):
        img = np.resize(img, (*FIXED_IMG_SIZE, 3))
    else:
        img = img.resize(FIXED_IMG_SIZE)
        img = np.asarray(img, dtype=np.float32)
    img_norm = total_comparisons * (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize to [0,21]
    return img_norm


def style_score(style1, style2):
    style_loss = np.mean(np.abs(style1 - style2))
    return style_loss  # low is good


def image_embeddings(vgg16, img):
    """
    Return embeddings of the given image
    """
    if isinstance(img, np.ndarray):
        img = np.resize(img, (1, 224, 224, 3))
    else:
        img = img.resize((224, 224))
        img = np.asarray(img, dtype=np.float32)
        if len(img.shape) == 2:
            # black and white sketch: add c = 3
            img = np.stack([img] * 3, axis=-1)
        img = np.expand_dims(img, axis=0) # (1, h, w, c)
    image_embedding = vgg16.predict(img)
    return image_embedding


def features_score(img1_embedding, img2_embedding):
    embeddings_similarity = cosine_similarity(img1_embedding, img2_embedding).reshape(1,)[0]
    return embeddings_similarity  # high is good


def image_preprocess_1(img):
    if isinstance(img, np.ndarray):
        img = np.resize(img, FIXED_IMG_SIZE)
        img = np.where(img > 0.5, 1.0, 0.0)
    else:
        img = img.resize(FIXED_IMG_SIZE)
        img = img.convert('1')  # black and white
        img = np.asarray(img, dtype=np.float32)
    return img


def similarity_score(img1, img2):
    ssim = skimage.metrics.structural_similarity(img1, img2, data_range=1.0)
    return ssim  # high is good


def generate_color_palette(img):
    img = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=FIXED_PALETTE_SIZE)
    palette = img.getpalette()  # RGB values
    palette = [palette[i:i + 3] for i in range(0, len(palette), 3)]
    palette = palette[0:FIXED_PALETTE_SIZE]
    palette = [(r / 255, g / 255, b / 255) for r, g, b in palette]  # normalize
    return palette


def visualize_palettes(generated_palette, reference_palette):
    _, ax = plt.subplots(figsize=(FIXED_PALETTE_SIZE + 1, 4))  # Extra space for labels
    ax.barh(y=1, width=[1] * FIXED_PALETTE_SIZE, left=range(FIXED_PALETTE_SIZE),
            color=generated_palette, edgecolor='none', height=0.6, align='center')
    ax.barh(y=0, width=[1] * FIXED_PALETTE_SIZE, left=range(FIXED_PALETTE_SIZE),
            color=reference_palette, edgecolor='none', height=0.6, align='center')
    ax.text(FIXED_PALETTE_SIZE + 0.5, 1, 'Generated', verticalalignment='center', fontsize=24)
    ax.text(FIXED_PALETTE_SIZE + 0.5, 0, 'Reference', verticalalignment='center', fontsize=24)
    ax.set_xlim(-0.5, FIXED_PALETTE_SIZE + 1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def palette_similarity(generated_palette, reference_palette):
    # transform RGB to CIELAB
    generated_palette = [RGB_to_LAB(rgb) for rgb in generated_palette]
    reference_palette = [RGB_to_LAB(rgb) for rgb in reference_palette]

    distances = cdist(generated_palette, reference_palette, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    similarity_score = np.mean(min_distances)
    return similarity_score  # low is good


def RGB_to_LAB(rgb):
    """
    The CIELAB color space is designed to be perceptually uniform,
    meaning that the Euclidean distance between two colors in this
    space more accurately reflects perceived differences.
    """
    return skimage.color.rgb2lab(rgb)


def true_alignment_score(generated, reference):
    """
    Given the generated image f(X_i) and the reference image E_i (the artist's real painting),
    measure the similarity of the generated image with the reference, in addition to
    a few independent false references and a few random guess references.

    The similarity is measured in terms of:
    1. structural (shapes and objects)
    2. style (gram matrix distance)
    3. features (VGG16 feature embeddings cosine similarity)
    4. color (palettes distance)
    """
    # structural
    structural_similarity_list = []
    generated_pre_ssim = image_preprocess_1(generated)
    reference_pre_ssim = image_preprocess_1(reference)
    ssim = similarity_score(generated_pre_ssim, reference_pre_ssim)
    print(f"ssim: {ssim}")
    structural_similarity_list.append(ssim)
    # style
    style_similarity_list = []
    generated_style = gram_matrix(generated)
    reference_style = gram_matrix(reference)
    style = style_score(generated_style, reference_style)
    print(f"style: {style}")
    style_similarity_list.append(style)

    # features
    features_similarity_list = []
    vgg16 = VGG16(weights='imagenet', include_top=False, 
                pooling='max', input_shape=(224, 224, 3))
    for model_layer in vgg16.layers:
        model_layer.trainable = False  # freeze for inference
    
    generated_embedding = image_embeddings(vgg16, generated)
    reference_embedding = image_embeddings(vgg16, reference)
    features = features_score(generated_embedding, reference_embedding)
    print(f"features: {features}")
    features_similarity_list.append(features)

    # color
    color_similarity_list = []
    generated_palette = generate_color_palette(generated)
    reference_palette = generate_color_palette(reference)
    visualize_palettes(generated_palette, reference_palette)
    palette = palette_similarity(generated_palette, reference_palette)
    print(f"palette: {palette}")
    color_similarity_list.append(palette)

    for i in range(num_of_ref_images):
        false_reference = Image.open(f"alignment_false_references/{i + 1}.jpeg")

        # ssim
        false_reference_pre_ssim = image_preprocess_1(false_reference)
        structural_similarity_list.append(similarity_score(generated_pre_ssim, false_reference_pre_ssim))
        
        # style
        false_reference_style = gram_matrix(false_reference)
        style_similarity_list.append(style_score(generated_style, false_reference_style))

        # features
        false_reference_embedding = image_embeddings(vgg16, false_reference)
        features_similarity_list.append(features_score(generated_embedding, false_reference_embedding))

        # color
        false_reference_palette = generate_color_palette(false_reference)
        color_similarity_list.append(palette_similarity(generated_palette, false_reference_palette))

    for i in range(num_of_ref_images):
        np.random.seed(i)  # freeze seed to generate the same random noise for every couple
        random_guess = np.random.random_sample(np.asarray(reference, dtype=np.float32).shape)

        # structural
        random_guess_pre_ssim = image_preprocess_1(random_guess)
        structural_similarity_list.append(similarity_score(generated_pre_ssim, random_guess_pre_ssim))

        # style
        random_guess_style = gram_matrix(random_guess)
        style_similarity_list.append(style_score(generated_style, random_guess_style))

        # features
        random_guess_embedding = image_embeddings(vgg16, random_guess)
        features_similarity_list.append(features_score(generated_embedding, random_guess_embedding))

        # color
        random_guess_palette = np.random.rand(FIXED_PALETTE_SIZE, 3)
        random_guess_palette = [tuple(row) for row in random_guess_palette]
        color_similarity_list.append(palette_similarity(generated_palette, random_guess_palette))

    # structural
    structural_similarity_sorted_indices = np.argsort(np.array(structural_similarity_list))
    structural_similarity_score = np.where(structural_similarity_sorted_indices == 0)[0][0] + 1

    # style
    style_similarity_sorted_indices = np.argsort(np.array(style_similarity_list))
    style_similarity_score = total_comparisons - np.where(style_similarity_sorted_indices == 0)[0][0]

    # features
    features_similarity_sorted_indices = np.argsort(np.array(features_similarity_list))
    features_similarity_score = np.where(features_similarity_sorted_indices == 0)[0][0] + 1

    # color
    color_similarity_sorted_indices = np.argsort(np.array(color_similarity_list))
    color_similarity_score = total_comparisons - np.where(color_similarity_sorted_indices == 0)[0][0]

    print(f"structural_similarity_score = {structural_similarity_score}")
    print(f"style_similarity_score = {style_similarity_score}")
    print(f"features_similarity_score = {features_similarity_score}")
    print(f"color_similarity_score = {color_similarity_score}")

    alignment_score = (structural_similarity_score + style_similarity_score + features_similarity_score + color_similarity_score) / (4 * total_comparisons)
    return alignment_score


def compute_alignment_score():
    table_path = "data.csv"
    df = pd.read_csv(table_path)
    a_list = []

    for index, row in df.iterrows():
        generated_path = row[('generated_artwork_name')]
        original_path = row[('image_path_named')]
        try:
            generated = Image.open(generated_path)
            reference = Image.open(original_path)
            a = true_alignment_score(generated, reference)
        except Exception as e:
            print(e)
            a = -1
        a_list.append((index, a))
        print(f"--- row {index} alignment_score = {a} ---")
        df.at[index, 'alignment_score'] = a
        df.to_csv(table_path, index=False)
    
    print(a_list)

#compute_alignment_score()

def single_sample(index):
    table_path = "data.csv"
    df = pd.read_csv(table_path)

    generated_path = df[('generated_artwork_name')][index]
    original_path = df[('image_path_named')][index]
    generated = Image.open(generated_path)
    reference = Image.open(original_path)
    a = true_alignment_score(generated, reference)
    print(f"--- row {index} alignment_score = {a} ---")

single_sample(663)