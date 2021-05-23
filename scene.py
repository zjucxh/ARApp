import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import common 
print(cv.__version__)
SCALE_FACTOR = 1
focal = 3351
img1 = cv.imread('corners/11.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('corners/12.jpg',cv.IMREAD_GRAYSCALE) # trainImage
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
    if m.distance < 0.6*n.distance:
        query.append(m.queryIdx)
        train.append(m.trainIdx)
        good.append([m])


# kp1[query[i]].pt is the location of key point
print(kp1[query[0]].pt)
print(kp2[train[0]].pt)
pt1 =np.array( [ [kp1[query[i]].pt[0],kp1[query[i]].pt[1]] for i in range(len(query))],dtype=np.float32)
pt2 =np.array( [ [kp2[train[i]].pt[0],kp2[train[i]].pt[1]] for i in range(len(train))],dtype=np.float32)
#print('pt1 = {0}'.format(pt1))
#print('pt2 = {0}'.format(pt2))

height, width = img1.shape
print('height, width = {0},{1}'.format(height, width))
cameraMatrix = np.array([[focal*SCALE_FACTOR, 0, 2019.4073],
                         [0.0, focal*SCALE_FACTOR, 1547.5577],
                         [0.0, 0.0, 1.0]],dtype=np.float32)
E, mask = cv.findEssentialMat(pt1, pt2, cameraMatrix=cameraMatrix,method=cv.RANSAC)

mask = mask.flatten()
inpt1 = np.array([pt1[i] for i in range(len(mask)) if mask[i]==1],dtype=np.float32)
inpt2 = np.array([pt2[i] for i in range(len(mask)) if mask[i]==1],dtype=np.float32)
#print("inpt2 = {0}".format(inpt2))
print("len inpt2 = {0}".format(len(inpt2)))
points, R,t,mask = cv.recoverPose(E, pt1, pt2)
print('recover points = {0}'.format(points))
print('R = {0}\n T = {1}'.format(R,t))

# TODO: triangulate 3d poitns
p1 = np.array([[1.0, 0.0, 0.0,0.0], 
               [0.0, 1.0, 0.0,0.0],
               [0.0, 0.0, 1.0,0.0]],dtype=np.float32)
p1 = np.dot(cameraMatrix,p1)
#p2 = K*[R|t]
p2 = np.append(R, np.dot(-R,t), axis=1)
p2 = np.dot(cameraMatrix,p2)

print('p2 = {0}'.format(p2))
points4d = cv.triangulatePoints(p1,p2,inpt1.T,inpt2.T)
print('points4d shape = {0}'.format(points4d.shape))
for i, e in enumerate(points4d):
    points4d[i] = e / points4d[-1]
points3d = points4d[:-1,:].T
print('point3d = {0}'.format(points3d[0]))
#TODO solve pnp
retval, rvec, tvec = cv.solvePnP(points3d,inpt2,cameraMatrix,None)
print('PNP: {0}, {1}, {2}'.format(retval, rvec, tvec))

print('project to image:')
point2d = common.project3dpt(cameraMatrix, rvec.flatten(), tvec.flatten(), points3d)
print(point2d[:,:-1])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()
