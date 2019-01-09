import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load two images
img1 = cv2.imread('pat.png')
img2 = cv2.imread('12.jpg')
(a,b,c) = img1.shape
print(type(a))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
rows, cols = img1.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
dst = cv2.warpAffine(img1, M, (cols, rows))
print(dst.shape)
cv2.imshow('dst',dst)
cv2.waitKey(0)

ret1,th1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
ret2,th2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret1)
print(ret2)
# print(ret1)
# cv2.imshow('1', th1)
# cv2.imshow('2', th2)
# cv2.waitKey(0)


# # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols ]
#
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# cv2.imshow('mask',mask)
# cv2.imshow('mask_inv',mask_inv)
#
#
# # Now black-out the area of logo in ROI
# img1_bg = cv2.bitwise_and(roi,img2,roi mask = mask)
#
# # Take only region of logo from logo image.
# img2_fg = cv2.bitwise_and(roi,img2,mask = mask)
# cv2.imshow('1',img1_bg)
# cv2.imshow('2',img2_fg)
# cv2.waitKey(0)
# # Put logo in ROI and modify the main image
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
#
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# e1 = cv2.getTickCount()
# print(e1)
# for i in range(5,49,2):
#     img1 = cv2.add(0,0)
# e2 = cv2.getTickCount()
# print(e2)
# print(cv2.getTickFrequency())
# t = (e2 - e1)/cv2.getTickFrequency()
# print(t)