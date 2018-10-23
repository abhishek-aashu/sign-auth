import math
import sys
import cv2
import numpy as np
import random
from collections import deque
import pickle

ch = int(input("\nDo you want to set the object first?\n\t1. Yes\n\t2. No\n\t"))
ocount = pickle.load(open("ocount.p", "rb"))

if(ch == 1):
    
    def nothing(x):
        pass

    cv2.namedWindow('image')

    cv2.createTrackbar('HMin','image',0,179,nothing)
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    cap = cv2.VideoCapture(0)

    waitTime = 330

    while(1):

        ret, img = cap.read()
        output = img

        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img,img, mask= mask)

        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax
            old_values = {"hMin": hMin, "sMin": sMin, "vMin": vMin, "hMax": hMax, "sMax": sMax, "vMax": vMax}  
            
        pickle.dump(old_values, open("old_values.p", "wb"))  

        output = cv2.flip(output,1)
        cv2.imshow('image',output)
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    
    old_values = pickle.load(open("old_values.p", "rb"))



n=1
while(n==1):
    
        
    cap = cv2.VideoCapture(0)
    
    center_points = deque()
      
    while True:
    
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)

        hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([old_values.get('hMin'), old_values.get('sMin'), old_values.get('vMin')])
        upper_blue = np.array([old_values.get('hMax'), old_values.get('sMax'), old_values.get('vMax')])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours) > 0:
            
            biggest_contour = max(contours, key=cv2.contourArea)

            moments = cv2.moments(biggest_contour)
            centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(frame, centre_of_contour, 5, (0, 0, 255), -1)

            ellipse = cv2.fitEllipse(biggest_contour)
            cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

            center_points.appendleft(centre_of_contour)

        for i in range(1, len(center_points)):
            b = random.randint(230, 255)
            g = random.randint(100, 255)
            r = random.randint(100, 255)
            if math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + (
                    (center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 50:
                cv2.line(frame, center_points[i - 1], center_points[i], (b, g, r), 4)
                cv2.line(mask, center_points[i - 1], center_points[i], (b, g, r), 4)

        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)

        filename = "outputs/sign_%d.jpg"%ocount
        cv2.imwrite(filename, mask)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        if k == 113:
            ocount+=1
            pickle.dump(ocount, open("ocount.p", "wb"))
            n = 2
            break

    
    cv2.destroyAllWindows()
    cap.release()
    
