

# A New Approach for Motion Prediction for Autonomous Driving

## Introduction

  In pursuit of advancing road safety, the automotive industry is rapidly evolving, with a particular focus on the development of autonomous vehicles. A paramount challenge in this endeavor is the ability to detect potential hazards and respond proactively to prevent accidents. This necessitates equipping autonomous vehicles with the capability to comprehend their dynamic environment and predict the movements of various traffic participants, including humans, vehicles, and motorcycles.
  
## Brief Analysis
  This repository introduces a novel approach to address this critical aspect of autonomous driving—vehicle motion prediction. Our methodology leverages a synergistic combination of Kalman Filter (predicted position, center of the car- blue color), the YOLO library (detected postion, center of the car-red color)and Clothoids (green color). 
  

<p align="center">
  <img src="https://github.com/PetrosAngelo/Route-Planning/blob/main/Screenshot/1stPic.png" alt="Simulation Results">
</p>

Given the positions of our object and by predicting its next positions at every moment, using the Kalman Filter, we can conclude which clothoid curve is the closest to the object's most possible route.

<p align="center">
  <img src="https://github.com/PetrosAngelo/Route-Planning/blob/main/Screenshot/3rdPic.png" alt="Simulation Results">
</p>

  The core objective of this project is to enhance the overall safety of autonomous vehicles by empowering them to make informed decisions based on a nuanced understanding of their surroundings. To validate the effectiveness of our approach, we calculate and extract the error of our estimations which is being declined to zero during the progress of the test.
