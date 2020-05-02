import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import collections

img = cv.imread('football_image.jpg', 1)
# roi=img[0:80,180:300]
# cv.rectangle(img,(180,0),(300,80),(255,0,255),1)
# roi=img[0:80,180:300]
# cv.imshow('image',img)
# cv.imshow('roi',roi)
# cv.waitKey(0)
# cv.destroyAllWindows()

row, col, channel = img.shape
data = img.reshape(row * col, channel)  # reshape as matrix[row*col,channel]

from sklearn.cluster import KMeans

K_cluster = KMeans(n_clusters=4)

K_cluster.fit(data)
label, freq = np.unique(K_cluster.labels_, return_counts=True)  # find frequency of each cluster
max_freq = np.argmax(freq)  # find index of maximum frequency

dom_label = label[max_freq]  # label corresponding to maximum frequency

dom_color = K_cluster.cluster_centers_[dom_label]  # cluster center pertaining to dominant color label
img[0:80, 270:400] = dom_color.astype('uint8')  # R, G, B value pertaining to dominant cluster center  (8 bit channel)
cv.putText(img, 'Dominant Color:', (10, 25), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
