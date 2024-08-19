import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage
from scipy.spatial.distance import cdist
import os

# Compute the alignment score A_i


FIXED_IMG_SIZE = (100,100)
FIXED_PALETTE_SIZE = 10

num_of_ref_images = os.listdir("alignment_false_references/")
num_of_ref_images = [file for file in num_of_ref_images if "jpeg" in file]
num_of_ref_images = len(num_of_ref_images)

def image_preprocess(img):
    """
    input: PIL Image object
    output: numpy array
    """
    img = img.resize(FIXED_IMG_SIZE)
    img = img.convert('1') # black and white
    img = np.asarray(img, dtype=np.float32)
    return img

def similarity_score(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2, data_range=1.0) # high is good

def similarity_alignment(generated, reference):
    ssim_list = []

    # true reference
    generated = image_preprocess(generated)
    reference = image_preprocess(reference)
    ssim_list.append(similarity_score(generated, reference))

    for i in range(num_of_ref_images):
        ref_img = Image.open(f"alignment_false_references/{i+1}.jpeg")
        ref_img = image_preprocess(ref_img)
        ssim = similarity_score(generated, ref_img)
        ssim_list.append(ssim)

    for i in range(num_of_ref_images):
        random_guess = np.random.random_sample(FIXED_IMG_SIZE)
        random_guess = np.where(random_guess > 0.5, 1.0, 0.0)
        ssim = similarity_score(generated, random_guess)
        ssim_list.append(ssim)

    return np.argmax(ssim_list) == 0

def generate_color_palette(img):
    img = img.convert('P', palette=Image.ADAPTIVE, colors=FIXED_PALETTE_SIZE)
    palette = img.getpalette() # RGB values
    palette = [palette[i:i+3] for i in range(0, len(palette), 3)]
    palette = palette[0:FIXED_PALETTE_SIZE]
    palette = [(r/255, g/255, b/255) for r, g, b in palette] # normalize
    return palette

def visualize_palettes(generated_palette, reference_palette):
    _, ax = plt.subplots(figsize=(FIXED_PALETTE_SIZE + 1, 4))  # Extra space for labels
    ax.barh(y=1, width=[1] * FIXED_PALETTE_SIZE, left=range(FIXED_PALETTE_SIZE),
            color=generated_palette, edgecolor='none', height=0.6, align='center')
    ax.barh(y=0, width=[1] * FIXED_PALETTE_SIZE, left=range(FIXED_PALETTE_SIZE),
            color=reference_palette, edgecolor='none', height=0.6, align='center')
    ax.text(FIXED_PALETTE_SIZE + 0.5, 1, 'Generated', verticalalignment='center', fontsize=12)
    ax.text(FIXED_PALETTE_SIZE + 0.5, 0, 'Reference', verticalalignment='center', fontsize=12)
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
    return similarity_score # low is good

def RGB_to_LAB(rgb):
    """
    The CIELAB color space is designed to be perceptually uniform,
    meaning that the Euclidean distance between two colors in this
    space more accurately reflects perceived differences.
    """
    return skimage.color.rgb2lab(rgb)

def color_alignment(generated, reference):
    pas_list = [] # palette similarity scores

    # true reference
    generated_palette = generate_color_palette(generated)
    reference_palette = generate_color_palette(reference)
    #visualize_palettes(generated_palette, reference_palette)
    pas_list.append(palette_similarity(generated_palette, reference_palette))

    for i in range(num_of_ref_images):
        ref_img = Image.open(f"alignment_false_references/{i+1}.jpeg")
        ref_pal = generate_color_palette(ref_img)
        pas = palette_similarity(generated_palette, ref_pal)
        pas_list.append(pas)

    for i in range(num_of_ref_images):
        random_guess = np.random.rand(FIXED_PALETTE_SIZE, 3)
        random_guess = [tuple(row) for row in random_guess]
        pas = palette_similarity(generated_palette, random_guess)
        pas_list.append(pas)

    return np.argmin(pas_list) == 0

def true_alignment_score(generated, reference):
    """
    Given the generated image f(X_i) and the reference image E_i (the artist's real painting),
    measure the similarity of the generated image with the reference, in addition to
    a few independent false references and a few random guess references.
    The similarity is measured once in terms of shapes and objects, and in color palettes.
    Only if the generated image is more similar to the ground truth reference
    rather than the other references, the function determines A_i = 1.
    """
    are_objects_similar = similarity_alignment(generated, reference)
    are_colors_similar = color_alignment(generated, reference)
    if are_objects_similar and are_colors_similar:
        return 1
    return 0


# Unit test
generated = Image.open('generated.jpeg')
reference = Image.open('reference.jpeg')
a = true_alignment_score(generated, reference)
assert a == 1


