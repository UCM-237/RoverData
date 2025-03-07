# This repository contains data extracted from Paparazzi UAV after performing experiments with our rovers.
Our rover's:
![Rover_MechTest1](https://github.com/user-attachments/assets/676bfb64-7307-4ea5-842d-02e25a82d640)
One of our rover's hardware:
![hardware_2](https://github.com/user-attachments/assets/ce4cd9eb-52a2-4025-8b59-04637e8a6c78)

## If you are coming from the 2025 CDC Article: Nonsmooth Guiding Vector Fields for Robot Motion and Control
* Experimental data is inside the folder [Data Folder](https://github.com/UCM-237/RoverData/tree/data_alfredo/data25Feb2025pNorm).
  * The .log and .data files can be used with Paparazzi UAV to do a replay of the experiments.
  * The .csv data files can be used to present the results of the experiments using, for example, python.
* From there you can find the notebook [dataPlot25Feb2025pNorm](https://github.com/UCM-237/RoverData/blob/data_alfredo/data25Feb2025pNorm/dataPlot25Feb2025pNorm.ipynb). 
  * It contains the mathematical expresions for the proximal normals of the presented trajectories in the article.
  * It contains a folder with the images presented in the article.
  * It also contains the data representation of an experiemnt where two vehicles follow concentric trajectories. These
    trajectories are codified as the boundary of the $p$ norm ball, with $p \in (1,2)$.
