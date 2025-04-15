import cv2
import numpy as np
from skimage import feature

def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute first order features
    mean = np.mean(image)
    variance = np.var(image)
    std_dev = np.std(image)
    skewness = feature.skimage.measure.compare_skew(image)
    kurtosis = feature.skimage.measure.compare_kurtosis(image)

    # Compute second order features
    glcm = feature.greycomatrix(image, [1], [0], symmetric=True, normed=True)
    contrast = feature.greycoprops(glcm, 'contrast')[0, 0]
    energy = feature.greycoprops(glcm, 'energy')[0, 0]
    asm = feature.greycoprops(glcm, 'ASM')[0, 0]
    entropy = -np.sum(glcm * np.log(glcm + 1e-10))  # Avoid log(0)
    homogeneity = feature.greycoprops(glcm, 'homogeneity')[0, 0]
    dissimilarity = feature.greycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = feature.greycoprops(glcm, 'correlation')[0, 0]
    coarseness = 1 / (1 + np.mean(image))

    # Combine features into a single array
    features = np.array([mean, variance, std_dev, skewness, kurtosis,
                         contrast, energy, asm, entropy, homogeneity,
                         dissimilarity, correlation, coarseness])
    return features
