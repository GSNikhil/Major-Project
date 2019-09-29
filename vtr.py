from sklearn.cluster import KMeans
import cv2
import numpy as np
import sys

image = cv2.imread("C:\Users\Amogh Kumar\Desktop\major project\sample7.jpg")
cv2.imshow("frame3",image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cluster=2

x = np.reshape(image,(image.shape[0]*image.shape[1],3))

clt = KMeans(n_clusters = cluster,n_init=4, random_state=5)
clt.fit(x)

centroids = clt.cluster_centers_
labels = clt.labels_

z=np.reshape(labels,(image.shape[0],image.shape[1]))
z = z.astype(np.uint8)
z=z*(255/cluster)
z = cv2.GaussianBlur(z, (1, 1), 0)
dst = cv2.Laplacian(z, cv2.CV_16S, ksize=3)
abs_dst = cv2.convertScaleAbs(dst)
cv2.imshow("frame2",abs_dst)
cv2.imshow("frame",z)
cv2.waitKey(0)
