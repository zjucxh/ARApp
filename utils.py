import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 


class Camera: 
    def __init__(self):
        self.intrinsic = None 
        self.rvec = np.zeros(shape=3,dtype=np.float64)
        self.tvec = np.zeros(shape=3,dtype=np.float64)
        self.rtom = np.eye(3,dtype=np.float64)

    
    def rotation_matrix(self):
        mat,_ = cv.Rodrigues(self.rvec)
        return mat

def reproject(camera, points3d):
    K = camera.intrinsic
    R = camera.rtom 
    T = camera.tvec 
    pt3d = points3d.T 
    pt2d = (np.dot(R, pt3d).T + T).T
    pt2d = np.dot(K, pt2d)
    pt2d = pt2d/pt2d[-1]
    pt2d = pt2d.T[:,:-1]
    return pt2d
   

def draw_point2d(img, points2d,num):
    for e in points2d:
        cv.circle(img, (int(e[0]),int(e[1])), 3, color=(255,0,0),thickness=3)
    #cv.namedWindow('image',cv.WINDOW_NORMAL)
    #cv.imshow('image', img)
    cv.imwrite('image{0}.jpg'.format(num), img)
    #cv.waitKey(0)

