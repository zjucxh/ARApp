import numpy as np 
import cv2 as cv 

def project3dpt(cameraMatrix, rvec, tvec, pt3d):
    rotm, _ = cv.Rodrigues(rvec)
    pt2d = []
    for e in pt3d:
        # image point = K*(R * X + t) * alpha
        pj = np.dot(cameraMatrix, np.dot(rotm,e)+tvec)
        pj = pj/pj[-1]
        pt2d = np.append(pt2d, pj)

    return pt2d.reshape(-1,3)

