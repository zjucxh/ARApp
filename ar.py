import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 
import open3d as o3d
import utils
from utils import Camera
from utils import reproject
import json 



class AR():
    def __init__(self):
        config = json.load(open('config.json'))
        self.image_left = cv.imread(config['image']['image_left'])
        self.image_right = cv.imread(config['image']['image_right'])
        self.camera_left = Camera()
        self.camera_right= Camera()
        focal = config['CameraIntrinsic']['focal']
        u0 = config['CameraIntrinsic']['u0']
        v0 = config['CameraIntrinsic']['v0']
        self.camera_left.intrinsic = np.array([[focal, 0, u0],[0, focal, v0],[0, 0, 1.0]], dtype=np.float64)
        self.camera_left.rtom = np.eye(3,dtype=np.float64)
        self.camera_right.intrinsic = np.array([[focal, 0, u0],[0, focal, v0],[0, 0, 1.0]], dtype=np.float64)
        self.camera_right.rtom = np.eye(3,dtype=np.float64)

    # Extract feature points from left and right images, match features
    # Return left and right point of feature point location on image
    def match_features(self):
        gray_left = cv.cvtColor(self.image_left,cv.COLOR_BGR2GRAY)
        gray_right= cv.cvtColor(self.image_right,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_left,None)
        kp2, des2 = sift.detectAndCompute(gray_right,None)
        self.kp1 = kp1
        self.kp2 = kp2
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        query = []
        train = []
        good = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                query.append(m.queryIdx)
                train.append(m.trainIdx)
                good.append([m])
        self.good_matches = good 

        left_pt =np.array( [ [kp1[query[i]].pt[0],kp1[query[i]].pt[1]] for i in range(len(query))],dtype=np.float64)
        right_pt=np.array( [ [kp2[train[i]].pt[0],kp2[train[i]].pt[1]] for i in range(len(train))],dtype=np.float64)
        return left_pt, right_pt
    # Draw good matching features
    def draw_good_matches(self):
        image = cv.drawMatchesKnn(self.image_left, self.kp1, self.image_right, self.kp2, self.good_matches,None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.namedWindow('matches',cv.WINDOW_NORMAL)
        cv.imshow('matches',image)
        cv.waitKey(0)
    def recover_pose(self):
        left_pt, right_pt = self.match_features()
        E, mask = cv.findEssentialMat(left_pt, right_pt, self.camera_left.intrinsic, cv.RANSAC)
        mask = mask.flatten()
        inlier_pt_left = np.array([left_pt[i] for i in range(len(mask)) if mask[i]==1],dtype=np.float64)
        inlier_pt_right = np.array([right_pt[i] for i in range(len(mask)) if mask[i]==1],dtype=np.float64)
        # recover relative camera pose
        # points = number of inliers
        points, R, t, mask = cv.recoverPose(E, inlier_pt_left, inlier_pt_right)
        self.inlier_pt_left = np.array([inlier_pt_left[i] for i in range(len(mask)) if mask[i]!=0],dtype=np.float64)
        self.inlier_pt_right = np.array([inlier_pt_right[i] for i in range(len(mask)) if mask[i]!=0],dtype=np.float64)
        self.camera_left.rtom = np.eye(3, dtype=np.float64)
        self.camera_left.tvec = np.zeros(3,dtype=np.float64)
        self.camera_right.rtom = np.array(R,dtype=np.float64)
        self.camera_right.tvec = np.array(t.flatten(),dtype=np.float64)
        print('camera extrinsic {0}, {1}'.format(self.camera_right.rtom, self.camera_right.tvec))
    
    # Estimate depth of inlier feature points
    def triangulate(self):
        t = np.reshape(self.camera_left.tvec, (3,1))
        P1 = np.append(self.camera_left.rtom,t, axis=1)
        P1 = np.dot(self.camera_left.intrinsic, P1)
        t = np.reshape(self.camera_right.tvec, (3,1))
        #t = np.dot(-self.camera_right.rtom, t)
        P2 = np.append(self.camera_right.rtom,t, axis=1)
        P2 = np.dot(self.camera_right.intrinsic, P2)
        print("projection = {0}, {1}".format(P1,P2))
        print('camera intrinsic :{0}'.format(self.camera_right.intrinsic))
        depth_point = cv.triangulatePoints(P1, P2, self.inlier_pt_left.T, self.inlier_pt_right.T)
        #print('depth_point = {0}'.format(depth_point))
        depth_point = depth_point / depth_point[-1]
        depth_point = depth_point[:-1].T
        self.depth_point = depth_point
        print('depth = {0}'.format(self.depth_point))


if __name__ == '__main__':
    
    ar = AR()
    #left_pt, right_pt = ar.match_features()
    
    #print('left_pt = {0}'.format(left_pt))
    #print('right_pt= {0}'.format(right_pt))
    #ar.draw_good_matches()
    ar.recover_pose()
    ar.triangulate()
    #draw left image projection
    pt2d = reproject(ar.camera_left, ar.depth_point)
    image = ar.image_left
    utils.draw_point2d(image, pt2d, 1)
    #Draw right image projection 
    pt2d = reproject(ar.camera_right, ar.depth_point)
    image = ar.image_right
    utils.draw_point2d(image, pt2d, 2)

