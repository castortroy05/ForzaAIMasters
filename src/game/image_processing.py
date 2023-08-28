import numpy as np

def preprocess_input_data(img):
    """
    Preprocess the input image data to extract relevant features.

    Parameters:
    - img (array-like): The input image.

    Returns:
    - features (np.array): The extracted features.
    """
    features = []
    if len(img.shape) == 3:  # If the img array is 3-dimensional
        for i in range(3):  # For each color channel (R, G, B)
            features.append(np.mean(img[:,:,i]))
            features.append(np.std(img[:,:,i]))
    elif len(img.shape) == 1:  # If the img array is 1-dimensional
        features.append(np.mean(img))
        features.append(np.std(img))
    else:
        raise ValueError(f"Unexpected shape of img array: {img.shape}")

    while len(features) < 6:
        features.append(0)
    return np.array(features)
