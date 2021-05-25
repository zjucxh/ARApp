import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d
import common 
print(cv.__version__)
print(o3d.__version__)
SCALE_FACTOR = 1
focal = 3351
img1 = cv.imread('corners/01.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('corners/02.jpg',cv.IMREAD_GRAYSCALE) # trainImage
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
inpt1 = np.append(inpt1, [[1811,2264]],axis=0)
inpt2 = np.append(inpt2, [[1518,2033]],axis=0)
print("inpt2 = {0}".format(inpt2))

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
p2 = np.append(R,np.dot(-R,t), axis=1)
p2 = np.dot(cameraMatrix,p2)

#print('p2 = {0}'.format(p2))
points4d = cv.triangulatePoints(p1,p2,inpt1.T,inpt2.T)
#print('points4d shape = {0}'.format(points4d.shape))

for i, e in enumerate(points4d):
    points4d[i] = e / points4d[-1]
points3d = points4d[:-1,:].T

print('point3d[-1] = {0}'.format(points3d[-1]))

## reproject points
common.project_axis(img1, p1, 1)
common.project_axis(img2, p2, 2)
#TODO solve pnp
#retval, rvec, tvec = cv.solvePnP(points3d,inpt1,cameraMatrix,None)
#print('rvec = {0}, tvec = {1}'.format(rvec, tvec))
#rotm, _ = cv.Rodrigues(rvec)
extrinsic1 = np.append(p1,[[0.0, 0.0, 0.0, 1.0]],axis=0)
extrinsic1 = [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
transform= [[1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]

extrinsic2 = np.append(R, np.dot(-R,t), axis=1)
extrinsic2 = np.append(extrinsic2, [[0.0, 0.0, 0.0, 1.0]],axis=0)
#extrinsic2 = np.dot(extrinsic2,transform)
#point2d = common.project3dpt(cameraMatrix, rvec.flatten(), tvec.flatten(), points3d)
#print(point2d[:,:-1])
#background = cv.imread('corners/1.jpg')
#common.project_axis(background,extrinsic1)
# load obj file and render scene
vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height)
mesh = o3d.io.read_triangle_mesh('corners/female.obj')
mesh.compute_vertex_normals()
mesh.translate([-0.0289, 0.1623, 9.117])
camera = o3d.camera.PinholeCameraParameters()
print('width = {0}, height = {1}'.format(width, height))
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focal, focal, 2019.0, 1547.0)
print('camera intrinsic = {0}'.format(intrinsic.intrinsic_matrix))
camera.intrinsic = intrinsic
#P = np.append(R, np.dot(-R,t),axis=1)

#print('P = {0}'.format(P))
camera.extrinsic = extrinsic1
#print('camera extrinsic1 = {0}'.format(extrinsic1))
# draw axis on image
#add geometry to image
vis.add_geometry(mesh)
image = o3d.io.read_image('corners/01.jpg')

vis.add_geometry(image)
ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(camera,True)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image('output/out1.png', True)
vis.run()
#vis.remove_geometry(mesh)

