import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

print(cv.__version__)

img1 = cv.imread('/home/chenxianghui/Pictures/img1.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('/home/chenxianghui/Pictures/img2.jpg',cv.IMREAD_GRAYSCALE) # trainImage
#initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
query = []
train = []
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        query.append(m.queryIdx)
        train.append(m.trainIdx)
        good.append([m])

#print(len(kp1))
#print(len(kp2))
print(kp1[query[0]].pt)
print(kp2[train[0]].pt)

print
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()
