import numpy as np
import cv2
im = cv2.imread('sample.jpg')
a=im.copy()
i=0
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(im, contours, -1, (0,255,0), 3)   
for cnt in contours:
    if cv2.contourArea(cnt)>2300 and cv2.contourArea(cnt)<1000000:
        x,y,w,h = cv2.boundingRect(cnt)
        z=a[y-50:y+h+50,x-50:x+w+50]
        cv2.imwrite(str(i)+".jpg",z)
        i+=1 
        cv2.imshow('z',z)
        cv2.waitKey(0)