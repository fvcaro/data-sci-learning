from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import warnings

warnings.filterwarnings('ignore')

def load_image(image_path):
    try:
        image = imread(image_path)
        print('Image loaded successfully ...')
        return image
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

def display_image(image, title='Image'):
    plt.imshow(image.astype(np.uint8))  # Convert to uint8 for proper display
    plt.title(title)
    plt.axis('off')
    plt.show()

def reshape_image(image):
    return image.reshape(-1, 3)

def segment_image(image, n_clusters=4, use_mini_batch=False, random_state=42):
    X = reshape_image(image)
    
    if use_mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        
    kmeans.fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)
    return segmented_img

def main(image_path, n_clusters=4, use_mini_batch=False):
    image = load_image(image_path)
    if image is not None:
        print('Image shape: ', image.shape)
        display_image(image, 'Original Image')
        
        segmented_image = segment_image(image, n_clusters, use_mini_batch)
        print('Segmented image shape: ', segmented_image.shape)
        display_image(segmented_image, f'Segmented Image with {n_clusters} Clusters')

# Example usage
if __name__ == "__main__":
    image_path = 'lady_bug.jpg'
    main(image_path, n_clusters=8, use_mini_batch=False)