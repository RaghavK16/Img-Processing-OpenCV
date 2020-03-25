import numpy as np
import cv2  # OpenCV library
import matplotlib.pyplot as plt

original_image = cv2.imread("sample.jpg")
# Converting from BGR Colours Space to HSV

img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1, 3))
# convert to np.float32

vectorized = np.float32(vectorized)
# Here we are applying k-means clustering so that the pixels around a colour are consistent and gave same BGR/HSV values
# define criteria, number of clusters(K) and apply kmeans()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# We are going to cluster with k = 2, because the image will have just two colours ,a white background and the colour
# of the patch

K = 5
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
# Now convert back into uint8
# now we have to access the labels to regenerate the clustered image


center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)
# res2 is the result of the frame which has undergone *k-means clustering*

figure_size = 15
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(res2)
plt.title('K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

# canny edge detection
edges = cv2.Canny(img, 100, 200)
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
