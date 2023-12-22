# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:40:42 2023

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

cap = cv2.VideoCapture("carfinal.mp4")

# Load detector
od = ObjectDetector()
# Load Kalman filter to predict the trajectory
kf = KalmanFilter()
results = []


angle_list = [-164.8270847, -160.6651915, -156.8014095, -151.7625545, -146.5751888, -141.0440922, -137.6630008, -136.3322199, -134.3259631, -135, -140.3145457,
-147.893744, -159.7751406, -175.2363583, 172.2781742, 159.4439548, 151.3138524, 152.5924246, 151.7625545, 150.8519282, 151.7625545, 151.7625545, 151.3138524,
153.0197693, 156.8014095, 160.6651915, 164.8270847, 171.326826, -179.0451587, -169.6111422, -156.9295878, -143.1301024, -134.3414568, -126.3044971, -119.9816394,
-116.5650512, -112.3286564, -106.3360429, -101.3099325, -95.71059314, -90.95484125, -87.13759477, -81.46923439, -74.82708465, -67.67134362, -60.48850144, -52.94347181,
-47.66300077, -41.63353934, -35.53767779, -32.90524292, -28.68614757, -25.30137863, -23.19859051, -20.22485943, -16.33604289, -12.42594287, -8.673174048, -1.877877447,
5.527540152, 14.2645123, 24.84238911, 34.21570213, 36.30449712, 33.42481118, 30.80144598, 29.51149856, 27.40757544, 22.32865638, 18.13808216, 14.2645123, 6.654425046,
0.954841254, -2.815556684, -8.53076561, -16.33604289, -23.19859051, -29.98163937, -36.30449712, -41.63353934, -45, -41.71075732, -34.21570213, -24.84238911, -16.60698058,
-8.53076561, -0.924045353, 8.53076561, 22.69379495, 31.293039, 38.95590784, 46.33221985, 51.04409216, 57.89374404, 65.55604522, 70.66519146, 73.66395711, 76.86597769,
80.53767779, 83.34557495, 88.09084757, 94.76364169, 99.46232221, 105.1729153, 111.4477363, 116.5650512, 121.6075022, 126.8698976, 131.0090869, 131.7107573, 133.6677801,
139.7078522, 147.0947571, 153.8531588, 159.7751406, 164.5778387, 169.4389893, 174.2894069, 179.0608091, -173.345575, -166.6512729, -163.0338519, -158.8951614, -154.2900462,
168.078732, 167.131913]

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
    results.append({ "Detected object's position": center,
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

