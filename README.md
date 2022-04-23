# Breast Cancer Prediction using FeedForward Neural Network
 
## 1. Summary
<p>The main objective of this project is to predict breast cancer result whether the tumour is malignant or benign.<br> 
 
The model is trained with [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). </p>

## 2. IDE and Framework
<p>This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, TensorFlow, Scikit-learn and Matplotlib. </p>

## 3. Methodology

### 3.1 Data Pipeline
<p>All data are loaded into IDE and all uwanted columns are removed. The labels are encoded by using OneHot label format. The data are then splitted into 80:20 train and test ratio. </p>

### 3.2 Model Pipeline
<p>A feedforward neural network is constructed that are specifically catered for binary classification problem. The architecture of the model is shown in figure below.<br>

 The model is trained with batch size of 32 and epochs of 25. After training, thea ccuracy obtained for the model is 99% with validation accuracy of 96%. </p>

## 4. Results

<p>The results from evaluating the model are shown below.</p>

![img1](https://user-images.githubusercontent.com/72061179/164891190-119ec5c5-1dc7-4498-b98e-18e9b351dc3a.png)
![img2](https://user-images.githubusercontent.com/72061179/164891193-cdc675b5-6618-455d-86a3-fe8b115be273.png)
![img3](https://user-images.githubusercontent.com/72061179/164891195-70fb7eb4-3c8f-4b19-b655-8a72ad046adb.png)
