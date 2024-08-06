from matplotlib import pyplot as plt
from matplotlib.image import imread
import warnings
from sklearn.cluster import KMeans

# Source: Pixel Perfect: Using K-Means Clustering to Segment Images Like a Pro in Python

warnings.filterwarnings('ignore')

image = imread('lady_bug.jpg')
print('image shape: ',image.shape)

plt.imshow(image)
plt.title('Lady bug')
plt.axis('off')
plt.show()

X = image.reshape(-1,3)
print(X.shape)

kmeans = KMeans(n_clusters = 3).fit(X)

print(kmeans.cluster_centers_)

print(kmeans.labels_)

segmented_image = kmeans.cluster_centers_[kmeans.labels_]

print(segmented_image)

segmented_image = segmented_image.reshape(image.shape)

# Ensure the pixel values are in the range [0, 1] for floats or [0, 255] for integers
if segmented_image.max() > 1:
    segmented_image = segmented_image / 255.0

print('Segmented image after reshaping and normalizing: ')
print(segmented_image)

# Display the segmented image
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')
plt.show()