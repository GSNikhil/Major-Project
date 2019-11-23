from sklearn.cluster import KMeans
import cv2
import numpy as np
import sys
import time
import math

image = cv2.imread("me3.jpg")
image_copy = np.copy(image)
shirt = cv2.imread("shirt3.jpg")

# image = cv2.resize(shirt, (480,840) )

cv2.imshow("Original Image",image)
cv2.imshow("Shirt", shirt)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cluster=2

x = np.reshape(rgb_image,(image.shape[0]*image.shape[1],3))

clt = KMeans(n_clusters = cluster,n_init=4, random_state=5)
clt.fit(x)

centroids = clt.cluster_centers_
labels = clt.labels_

z=np.reshape(labels,(image.shape[0],image.shape[1]))
z = z.astype(np.uint8)
z = z*255
z = cv2.GaussianBlur(z, (1, 1), 0)
dst = cv2.Laplacian(z, cv2.CV_16S, ksize=3)
edge = cv2.convertScaleAbs(dst)
#np.set_printoptions(threshold=sys.maxsize)
#print(edge)

MODE = "MPI"

if MODE is "COCO":
    protoFile = "./OpenPose/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "./OpenPose/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "./OpenPose/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./OpenPose/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


# frame = cv2.imread("model.jpg")
frameCopy = np.copy(image)
frameWidth = image.shape[1]
frameHeight = image.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

t = time.time()
# input image dimensions for the network
inWidth = frameWidth
inHeight = frameHeight
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold : 
        #cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
        # print(str(x) + " " + str(y))
        # print(str(len(points)))
    else :
        points.append(None)

# Draw Skeleton
# for pair in POSE_PAIRS:
#     partA = pair[0]
#     partB = pair[1]

#     if points[partA] and points[partB]:
#         cv2.line(image, points[partA], points[partB], (0, 255, 255), 2)
#         cv2.circle(image, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)



print("Total time taken : {:.3f}".format(time.time() - t))

left_hip = np.array(points[8])
right_hip = np.array(points[11])
left_shoulder = np.array(points[2])
right_shoulder = np.array(points[5])



#Left-Hip
while left_hip[0] > 0:
    if edge[left_hip[1]][left_hip[0]] == 255:
        print(left_hip)
        break;

    left_hip[0] = left_hip[0] - 1;

#Right-Hip
while right_hip[0] < edge.shape[1]:
    if edge[right_hip[1]][right_hip[0]] == 255:
        print(right_hip)
        break;
    
    right_hip[0] = right_hip[0] + 1;
    # print(edge[right_hip[0]][right_hip[1]])

cv2.circle(image, tuple(left_hip), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
cv2.circle(image, tuple(right_hip), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
cv2.circle(image, tuple(left_shoulder), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)
cv2.circle(image, tuple(right_shoulder), 8, (255, 0, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow("K-Means Output",z)
cv2.imshow("Edge Detection", edge)
cv2.imshow("With Dots",image)

shirt_left_shoulder = [72, 43]
shirt_right_shoulder = [315,43]
shirt_bottom = [85, 400]

x_shirt = shirt_right_shoulder[0] - shirt_left_shoulder[0]
y_shirt = shirt_bottom[1] - shirt_left_shoulder[1]

x_person = right_shoulder[0] - left_shoulder[0]
y_person = left_hip[1] - left_shoulder[1]

x_scale = x_person * 1.0 /  x_shirt *1.1
y_scale = y_person * 1.0 / y_shirt  *1.1


shirt = cv2.resize(shirt, None,fx=x_scale,fy=y_scale)
shirt_left_shoulder_resize=[math.floor(shirt_left_shoulder[0]*x_scale),math.floor(shirt_left_shoulder[1]*y_scale)]
print(shirt_left_shoulder[0]/x_scale)
print(shirt_left_shoulder[1]/y_scale)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = shirt.shape

x_offset = left_shoulder[0]-shirt_left_shoulder_resize[0] 
y_offset = left_shoulder[1]-shirt_left_shoulder_resize[1] - 10

# roi = image[x_offset: x_offset + rows, y_offset : y_offset + cols]
roi = image_copy[y_offset : y_offset + rows, x_offset : x_offset + cols]
# print(left_shoulder[0]-shirt_left_shoulder_resize[0], left_shoulder[1]-shirt_left_shoulder_resize[1])
# Now create a mask of logo and create its inverse mask
img2gray = cv2.cvtColor(shirt,cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', img2gray)

# add a threshold
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Mask", mask)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow("Mask Inverted", mask_inv)

# # Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
cv2.imshow("Background", img1_bg)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(shirt,shirt,mask = mask)
cv2.imshow("Foreground", img2_fg)

dst = cv2.add(img1_bg,img2_fg)

# image[left_shoulder[1]-shirt_left_shoulder_resize[1]:left_shoulder[1]-shirt_left_shoulder_resize[1]+cols,left_shoulder[0]-shirt_left_shoulder_resize[0]:left_shoulder[0]-shirt_left_shoulder_resize[0]+rows ] = dst
# image[x_offset: x_offset + rows, y_offset : y_offset + cols] = dst
image_copy[y_offset : y_offset + rows, x_offset : x_offset + cols] = dst

cv2.imshow('Final Image', image_copy)

cv2.waitKey(0)