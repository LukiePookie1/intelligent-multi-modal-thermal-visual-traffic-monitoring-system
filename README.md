# Intelligent Multi-Modal Visual-Thermal Traffic Monitoring System

**Computer Science Capstone Project for LARRI:**

This system is designed to intelligently monitor traffic by detecting and classifying vehicles and analyzing congestion patterns using multi-modal visual and thermal imaging.

**Team 7:** Thermal and Visual Imaging with Drones

**Sponsor:** Dr. Sabur Baidya

## Equipment

- HikMicro Pocket Series Thermal Imaging Camera
- DJI Phantom 4 PRO Professional Drone

## Dependencies for Object Detection and Image Classification

- Python 3.9 or higher
- NumPy
- Pandas
- Matplotlib
- OpenCV (cv2)
- PyTorch
- Torchvision
- Pillow (PIL)
- CUDA 11.2 (required for dedicated GPU setups)
- tqdm
- scipy

## Contributors

**Software Development:**
- Luke Rappa
- Morgan Taylor

**Hardware Integration:**
- Lucas Arvin
- Kevin Nguyen

## Directory Structure

- `annotations`: Contains XML files organized into folders based on what was annotated.
- `output`: Automatically generated upon program execution; includes timestamped folders with new videos and reports.
- `proto`: Prototype code; includes a variety of tests and experiments not used in the final product but still reusable.
- `quick-testing`: Contains folders with small images for classification and short videos for object detection testing.
- `raw-images`: Sliced images from traffic footage, used for training models.
- `raw-traffic-footage`: Contains the raw MP4 videos collected from traffic data.

## Scripts

- `vehicle-analysis.py`: Main script for interfacing with the system, offering options for classification or object detection.
  ```bash
  python vehicle-analysis.py

## Directions

- Clone the directory

- Install Python

- Install any nesseisary dependencies (likely Pytorch, CV2, scipy, and tqdm.)

- The project can be started by running vehicle-analysis.py and following the on-screen instructions. It will give you the option to use image classifcation or object detection. You can also run the vehicle classification and object detection individually. 
