import numpy as np
import apriltag
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv.imread('tag41_12_00000.png',0)          # Tag image
#img2 = cv.imread('TB_homtest2.png',0) # trainImage
img2 = cv.imread('data/01/0-002-RGBWarped.png',0) # trainImage

# Scale tag image
scale_percent = 40 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv.resize(img1, dim, interpolation = cv.INTER_AREA)

#Threshold trainimage?
#img2 = cv.bilateralFilter(img2,9,75,75)
#img2 = cv.adaptiveThreshold(img2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

# Initiate SIFT detector
#sift = cv.SIFT_create()
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

#Debug
print("IMG1 Keypoints & Descriptors:")
print("kp1 - ",type(kp1), " Len:",len(kp1))
print("des1 - ",type(des1), " Len:",len(des1))
print("IMG2 Keypoints & Descriptors:")
print("kp2 - ",type(kp2), " Len:",len(kp2))
print("des2 - ",type(des2), " Len:",len(des2))

# BF Matcher
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
print("BF:",type(bf))
matches = bf.match(des1,des2)
print("MATCHES:",type(matches), " Len:",len(matches))
print("\t\t",matches[:5])
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:],None, flags=2)



#Debug
dm = matches[0]
print("DM0:",dm.distance,dm.trainIdx,dm.queryIdx,dm.imgIdx)
print("Train:",des2[dm.trainIdx])
print("Query:",des1[dm.queryIdx])
dm = matches[-1]
print("DM-1:",dm.distance,dm.trainIdx,dm.queryIdx,dm.imgIdx)
print("Train:",des2[dm.trainIdx])
print("Query:",des1[dm.queryIdx])


img1b = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2b = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

cv.imshow('img1', img1b)
cv.imshow('img2', img2b)
plt.imshow(img3)
plt.show()
#cv.waitKey()
