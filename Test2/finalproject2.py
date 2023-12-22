# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:45:26 2023

@author: petro
"""

import numpy as np
import cv2
from scipy.integrate import odeint
import pandas as pd
import keyboard
import math

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return np.array([x, y])
        
class ObjectDetector:
    def __init__(self):
        # Create mask for orange color
        self.low_blue = np.array([11, 128, 90])
        self.high_blue = np.array([179, 255, 255])
        

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks with color ranges
        mask = cv2.inRange(hsv_img, self.low_blue, self.high_blue)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            break

        return box
    
def clothoid_ode_rhs(state, s, kappa0, kappa1):
    x, y, theta = state[0], state[1], state[2]
    return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1*s])

def eval_clothoid(x0, y0, theta0, kappa0, kappa1, s):
    return odeint(clothoid_ode_rhs, np.array([x0, y0, theta0]), s, (kappa0, kappa1))

cap = cv2.VideoCapture("carfinal2.mp4")

# Load detector
od = ObjectDetector()
# Load Kalman filter to predict the trajectory
kf = KalmanFilter()
results = []


angle_list = [-180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180,
-180, -180, -180, -180, -178.6360725, -176.0090869, -172.0565282, -166.293039, -160.7099538, -153.4349488, -146.6893692, -140.7105931, -134.0608091, -126.326826,
-120.7354877, -115.974394, -108.8531588, -103.706961, -98.13010235, -92.66300077, -88.63607247, -84.68545433, -79.21570213, -73.68614757, -69.44395478, -63.43494882,
-57.42594287, -51.76617482, -45, -37.40535663, -32.57405713, -26.56505118, -19.29004622, -13.706961, -6.788974574, -1.332219854, 0, 0, 1.332219854, 0, -1.363927532,
1.332219854, 0, -1.332219854, 0, 0, 0, 1.332219854, 0, -1.332219854,1.332219854, 0, -1.332219854, 0, 0, 0, 1.332219854, 0, -1.332219854, 1.332219854, 0, 0, 0,
-1.332219854, 1.332219854, 2.663000766, 6.788974574, 13.706961, 19.29004622, 27.18111109, 33.69006753, 39.28940686, 46.90915243, 53.39292519, 60.42216132,
66.44773633, 70.70995378, 76.60750225, 81.86989765, 85.91438322, 91.33221985, 96.78897457, 100.7842979, 105.0183606, 111.037511, 115.974394, 120.7354877, 128.2338252,
133.0908476, 140.7105931, 148.1340223, 153.4349488, 161.9958384, 169.2157021, 173.2110254, 176.4236656, -180, 166.676592, 166.4609344]

frame_count = 0

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    
    bluecar_box = od.detect(frame)
    x, y, w, h = bluecar_box
    cx = int((x + w) / 2) 
    cy = int((y + h) / 2)
    center = (cx, cy)

    predicted = kf.predict(cx, cy)
    x, y = predicted
   
    clothoid_curves = []
    angle = (angle_list[frame_count % len(angle_list)])
    closest_point = None
    min_distance = float('inf')
    error = np.sqrt((center[0] - predicted[0])**2 + (center[1] - predicted[1])**2)
    error_percent = error / np.sqrt(predicted[0]**2 + predicted[1]**2) * 100

    # Define scaling factor and offset to center the clothoid on the image
    scale_factor = 10
    
    for theta0 in np.arange(-0.3, 0.31, 0.15):
        x0, y0 = 0,0
        L = 6
        kappa0 = 0
        
        if theta0 < 0:
         kappa1 = -0.03
        elif theta0 > 0:
            kappa1 = 0.03
        else:
            kappa1 = 0
        s = np.linspace(0, L, 1000)
        sol = eval_clothoid(x0, y0, theta0, kappa0, kappa1, s)
        xs, ys, thetas = sol[:, 0], sol[:, 1], sol[:, 2]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_x_i = xs * M[0][0] +  ys * M[0][1] + M[0][2]
        rotated_y_i = xs * M[1][0] +  ys * M[1][1] + M[1][2]
        for i in range(len(xs) - 1):
            # Scale and shift coordinates to fit the image
            pt1 = np.array([int(xs[i]*scale_factor + cx), int(ys[i]*scale_factor + cy)])
            pt2 = np.array([int(xs[i+1]*scale_factor + cx), int(ys[i+1]*scale_factor + cy)])
            # Rotate clothoid curve
            pts = np.vstack([pt1, pt2]).T
            pts = np.vstack([pts, np.ones(pts.shape[1])])
            pts = M.dot(pts).T[:, :2]
            pt1_rotated, pt2_rotated = tuple(pts[0]), tuple(pts[1])
            pt1_rotated = tuple(map(int, pt1_rotated))
            pt2_rotated = tuple(map(int, pt2_rotated))
            # Draw a line between the two points
            cv2.line(frame, pt1_rotated, pt2_rotated, (0, 255, 0), 1)
            dist = np.sqrt((pt1_rotated[0]-predicted[0])**2 + (pt1_rotated[1]-predicted[1])**2)
            if dist < min_distance:
                min_distance = dist
                closest_point = pt1_rotated
                
    # Draw a red circle around the point with minimum distance, if a curve was found
    if closest_point is not None:
                cv2.circle(frame, closest_point, 3, (0, 0, 255), -1)
       
    frame_count += 1  # increment frame_count at the end of each loop iteration
       
                
    
    print(f"Detected object's position': {center}")
    print(f"Predicted postion: {predicted}")
    print(f"Angle: {angle}")
    results.append({ "Detected object's position": center,"cx": cx,"cy": cy,"predicted[0]": predicted[0], "predicted[1]": predicted[1], "Error": error_percent,
                             'Predicted position': predicted,'Angle': angle})
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    cv2.circle(frame, (predicted[0], predicted[1]), 3, (255, 0, 0), -1)
    
    
    
   

# Create a DataFrame from the list
    df = pd.DataFrame(results)

# Write the dataframe to an excel file
    df.to_excel(r'C:\Users\petro\Desktop\output.xlsx', index=False)

    
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(150)
    if keyboard.is_pressed('p'):
        print("Program paused")
        keyboard.wait('p')  # Wait until 'p' is released
        print("Program resumed")   
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
