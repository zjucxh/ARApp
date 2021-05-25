import numpy as np 
import cv2 as cv 


def reproject(img, P, point4d):
    image_points = np.dot(P, point4d)
    image_points = image_points/image_points[-1] 
    #print('image_points = {0}'.format(image_points))
    x = image_points[0]
    y = image_points[1]
    for i in range(len(x)):
        cv.circle(img, (int(x[i]), int(y[i])),2,(255,0,0),thickness=3)
    cv.imwrite('keypoints2.jpg',img)

def project_axis(img, P, num):
    print('P = {0}'.format(P))
    point4d = np.array([[0.0, 0.0, 10.0, 1.0],
                        [1.0, 0.0, 10.0, 1.0],
                        [0.0, 1.0, 10.0, 1.0],
                        [0.0, 0.0, 11.0, 1.0]],dtype=np.float32)
    image_points = np.dot(P, point4d.T)
    image_points = image_points/image_points[-1] 
    x = image_points[0]
    y = image_points[1]
    print('image_points = {0}'.format(image_points))
    for i in range(len(x)):
        cv.line(img,(int(x[0]), int(y[0])),(int(x[i]), int(y[i])),color=(255,0,0),thickness=9)
    cv.imwrite('axis{0}.jpg'.format(num),img)

    
