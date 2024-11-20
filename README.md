# Distracted and Drowsy Driver Detection

## Project Overview

Distracted driving is a major cause of road accidents. According to the CDC Motor Vehicle Safety Division, one in five car accidents is caused by a distracted driver. This translates to 4,25,000 people injured and 3,000 lives lost annually.

In this project, machine learning models were developed and refined to detect distracted driving behaviors based on driver images. Additionally, the project includes **drowsiness detection**, leveraging facial landmarks to identify signs of fatigue.

These models aim to enhance road safety by identifying dangerous behaviors such as texting, eating, or yawning in real-time.

---

## Problem Statement

Given a dataset of 2D dashboard camera images, an algorithm is needed to:

1. Classify a driver's behavior to determine if they are driving attentively or engaging in distracted behaviors (e.g., texting, talking on the phone, operating the radio, etc.).
2. Detect signs of **drowsiness**, including yawning and prolonged eye closure, using facial landmark detection.

This solution can be deployed with dashboard cameras to automatically monitor driver behavior and issue alerts for potential hazards.

---

## Tasks

The project involves the following tasks:

1. **Data Download and Preprocessing**:
   - Obtain and preprocess driver images for training, validation, and testing.
2. **Model Development**:
   - Build and train a machine learning model to classify distracted driving behaviors.
   - Implement drowsiness detection using the `shape_predictor_68_face_landmarks` model to track key facial points.
3. **Model Testing and Optimization**:
   - Test the models for accuracy and robustness.
   - Improve performance using advanced techniques such as data augmentation, hyperparameter tuning, and transfer learning.

---

## Data Exploration

The dataset used for this project is obtained from Kaggle's _State Farm Distracted Driver Detection_ competition. It includes images of drivers engaged in various activities.

### File Descriptions

- **`imgs.zip`**: Zipped folder containing all images (train/test).
- **`sample_submission.csv`**: A sample submission file in the correct format.
- **`driver_imgs_list.csv`**: A CSV file listing training images, driver IDs, and class labels.

The images are color images with a resolution of 640x480 pixels.

### Behavior Categories (Classes)

The dataset contains the following 10 classes:

1. **c0**: Safe driving
2. **c1**: Texting - right
3. **c2**: Talking on the phone - right
4. **c3**: Texting - left
5. **c4**: Talking on the phone - left
6. **c5**: Operating the radio
7. **c6**: Drinking
8. **c7**: Reaching behind
9. **c8**: Hair and makeup
10. **c9**: Talking to a passenger

### Dataset Statistics

- **Total Images**: 102,150
  - **Training Images**: 79,726
  - **Validation Images**: 17,939
  - **Test Images**: 4,485

---

## Drowsiness Detection

To enhance the project's capabilities, a **drowsiness detection feature** was added using the `shape_predictor_68_face_landmarks` model. This model tracks facial landmarks, enabling real-time detection of yawning and prolonged eye closure.

### Methodology

1. **Yawning Detection**:

   - Uses mouth-related facial points.
   - If the distance between specific lip points exceeds a threshold, the system detects yawning.

2. **Eye Closure Detection**:
   - Monitors eye-related facial points.
   - If the distance between eyelid points falls below a threshold for a sustained period, drowsiness is identified.

### Output

When signs of drowsiness (e.g., yawning or prolonged eye closure) are detected, alerts can be triggered to warn the driver.

## Run Locally

Clone the Project

```bash
  git clone https://github.com/Lava-Kumar-PL/DriverStateDetction_And_DrosinessDetection.git
```

Download the dataset from:-
[Kaggle State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

Extract the dataset from the zip file and move the test and train dataset into

```
Training Notebooks\imgs
```

### Prerequisites

- Python 3.x
- Required libraries:
  ```bash
  pip install -r requirements.txt
  ```
  Run `Training Notebooks\CustomCNN.ipynb` file to build the CNN Model which detects the Driver's Distracted state and the best model will saved locally in `Training Notebooks\model\self_trained`.And note down the output in the cell **Converting into numerical values**

Move the best Model from `Training Notebooks\model\self_trained` to `driverStateDetection\`

In the `driverStateDetection\app.py` in variable **maping** enter your **Converting into numerical values** by changing the keys to values and values to keys

Go to the project directory

```bash
  cd driverStateDetection
```

Start the driver state detection app

```bash
  python app.py
```

Open new terminal

Activate virtual environment as the libraries version required to read 'shape_predictor_68_face_landmarks` needs an older version which conflicts with the version required by the Driver state detection app

Anaconda should be installed locally

```bash
  conda activate env
```

Change the directory

```bash
  cd DrosinessApplication
```

Run DrosinessApplication

```bash
  python app.py
```
