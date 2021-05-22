import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('corners/*.jpg')

for fname in images:
    print('processing {0}'.format(fname))
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    #find the chess board corners
    print('finding chess board corners')
    ret, corners = cv.findChessboardCorners(gray,(7,9),None)
    print('found!')
    #If found, add object points, image points(after refining them)
    if ret == True:
        print('chess board corner found')
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,9), corners2, ret)
        cv.namedWindow('img',cv.WINDOW_NORMAL)
        cv.imshow('img',img)
        cv.waitKey(0)

cv.destroyAllWindows()
#print('objponts, imgpoints = {0}, {1}'.format(objpoints, imgpoints))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#rotm1, _ = cv.Rodrigues(rvecs[0])
#rotm2, _ = cv.Rodrigues(rvecs[1])
print('rvecs={0}'.format(rvecs))
print('tvecs = {0}'.format(tvecs))

'''
rvecs=[[ 0.00427284 -0.95528867 -0.29564387]
 [ 0.96252031  0.08410186 -0.25784014]
 [ 0.27117596 -0.28346152  0.9198441 ]]
rvecs=[[ 3.09238120e-03 -9.99975894e-01  6.21681446e-03]
 [ 8.37976158e-01 -8.01232140e-04 -5.45706255e-01]
 [ 5.45698082e-01  6.89707405e-03  8.37953480e-01]]
tvecs = [array([[ 4.0444291 ],
       [-3.92959324],
       [26.74290568]]), array([[ 2.69250133],
       [-2.89345395],
       [27.59065277]])]

'''

