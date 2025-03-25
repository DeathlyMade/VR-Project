# Mask Detection and Segmentation Project

## 1. Introduction
This project focuses on detecting and segmenting face masks in images using various machine learning and deep learning techniques. The tasks include binary classification using handcrafted features and machine learning classifiers, binary classification using CNNs, region segmentation using traditional techniques, and mask segmentation using U-Net. The objective is to evaluate the performance of different approaches in terms of classification accuracy and segmentation quality.

### Contributors:

(IMT2022022) Nathan <Nathan.Verghese@iiitb.ac.in>

(IMT2022010) Divyam <Divyam.Sareen@iiitb.ac.in>

(IMT2022521) Sarvesh <SarveshKumar.A@iiitb.ac.in>

---

## 2. Dataset
### Source:
https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset

https://github.com/sadjadrz/MFSD

```
MSFD
├── 1
│   ├── face_crop # face-cropped images of images in MSFD/1/img
│   ├── face_crop_segmentation # ground truth of segmend face-mask
│   └── img
└── 2
    └── img



dataset
├── with_mask # contains images with mask
└── without_mask # contains images without face-mask
```

### Structure:
- *Training Set:* Images used for training the models.
- *Testing Set:* Images used to evaluate model performance.
- *Annotations:* Mask region labels for segmentation tasks.

---

## 3. Objectives

### a. Binary Classification Using Handcrafted Features and ML Classifiers
1. Extract handcrafted features from facial images (Histogram of Oriented Gradients).
2. Train and evaluate at least two machine learning classifiers (We use XGBoost, Neural Network and SVM) and compare classifier performances based on accuracy.

### b. Binary Classification Using CNN
1. Design and train a Convolutional Neural Network (CNN) for mask classification.
2. Experiment with different hyperparameters (learning rate, batch size, optimizer, activation function).
3. Compare CNN performance with traditional ML classifiers.

### c. Region Segmentation Using Traditional Techniques
1. Apply region-based segmentation methods (e.g., thresholding, edge detection) to segment mask regions, and visualize and evaluate segmentation results.

### d. Mask Segmentation Using U-Net
1. Train a U-Net model to segment the mask regions in facial images.
2. Compare segmentation performance with traditional techniques using IoU or Dice score.

---

## 4. Hyperparameters and Experiments


### Hyperparameters Used in Part B

#### *Model Architecture Hyperparameters*  
| Hyperparameter         | Value              | Description |
|------------------------|-------------------|-------------|
| *Input Image Size*   | (150, 150, 3)     | Each image is resized to *150×150* with *3 channels (RGB)*. |
| *Number of Conv Layers* | 3 | Extracts hierarchical features. |
| *Filter Sizes*       | (3,3) | Kernel size used in Conv layers. |
| *Number of Filters per Layer* | 32 → 64 → 128 | Increasing filters to capture more complex features. |
| *Pooling Layer*      | MaxPooling2D (2×2) | Reduces spatial dimensions. |
| *Activation Function* | ReLU | Introduces non-linearity. |
| *Fully Connected Layer* | Dense(128, activation='relu') | Fully connected hidden layer. |
| *Dropout Rate*       | 0.5 | Prevents overfitting. |
| *Output Layer*       | Dense(1, activation='sigmoid') | Binary classification (mask/no-mask). |

---

#### *Training Hyperparameters*  
| Hyperparameter  | Value | Description |
|----------------|-------|-------------|
| *Optimizer*  | Adam, SGD, RMSprop  |  |
| *Loss Function* | Binary Crossentropy and Crossentropy | Suitable for binary classification. |
| *Metrics*    | Accuracy | Evaluates performance. |
| *Epochs*     | 30 (with Early Stopping) | Number of times model sees the dataset. |
| *Batch Size* | 32/64 | Number of images per batch. |

---

#### *Dataset Splitting Hyperparameters*  
| Hyperparameter  | Value  | Description |
|----------------|--------|-------------|
| *Train Ratio* | 70% | Data used for training. |
| *Validation Ratio* | 15% | Used for tuning hyperparameters. |
| *Test Ratio* | 15% | Used for final model evaluation. |

---

## 5. Results
### Evaluation Metrics:
- *Classification:* Accuracy, Precision, Recall, F1-score
- *Segmentation:* Intersection over Union (IoU), Dice Score

| Model | Accuracy (%) | IoU | Dice Score |
|--------|------------|----|-----------|
| XGBoost (part a) | 90.76% (80-20 train-test split with augmentations) | - | - |
| Neural Network (part a)| 89.05% (80-20 train-test split with augmentations) | - | - |
| SVM (part a)| 92.51% (80-20 train-test split with augmentations) | - | - |
| CNN (part b) | 96.74% Test Accuracy (70-15-15 train-validation-test split)| - | - |
| K-mean clustering | - | 0.554 (mean over 1st 10 images) | 0.414 (mean over 1st 10 images)
| Otsu's Threshold | - | 0.545 (mean over 1st 10 images)|0.404 (mean over 1st 10 images)
| Region-growing| - | 0.129 (mean over 1st 10 images) | 0.082 (mean over 1st 10 images)|
| U-Net Segmentation | 96.64% |91.37% | 95.09% |

---

## 6. Observations and Analysis

### PART A

For each image here we need to make a feature vector. We choose 5 features: color features, HoG, Edge features, texture featuresand ORB fetaures. 

**Since feature vector coresponding to images may be of diffrent lentgh, we resize all image and fix the length of individual sub-feature vectors, so that `np.hstack() `can work without interrupts when all individual sub-feature vectors ar combined into one vector for an image. Data used is dataset. We train an XGBoost model as well as a neural network and as observed, the test accuracy of XGBoost is better. This is attributed to the fact that neural networks need a lot of data to learn and here we have 4095 images.

### PART B


### Project Overview
This project implements a *binary classification* task using a *Convolutional Neural Network (CNN)* to detect whether a person is wearing a face mask or not. The dataset is split into training, validation, and testing sets. Various *hyperparameters* such as batch size, learning rate, optimizer, and activation functions were experimented with to achieve the best performance. The CNN model's performance is then compared with traditional *Machine Learning (ML) classifiers*.

---

### Dataset Preparation
#### Dataset Splitting
- The original dataset consists of two classes: with_mask and without_mask.
- Images are split into *70% training, 15% validation, and 15% testing*.

---

### Model Architecture
A *CNN model* is designed with the following layers:
1. *Convolutional Layers: Extract spatial features using **ReLU activation*.
2. *MaxPooling Layers*: Reduce spatial dimensions to prevent overfitting.
3. *Flatten Layer*: Converts feature maps into a single vector.
4. *Fully Connected (Dense) Layer*: Learns classification features.
5. *Dropout Layer (0.5)*: Reduces overfitting.
6. *Output Layer (Sigmoid Activation)*: Outputs a probability score for binary classification.

---

### Training and Hyperparameter Tuning
- *Batch Size:* 32/64
- *Epochs:* 30 
- *Optimizer:* Adam / SGD / RMSprop
- *Loss Function:* Binary Cross-Entropy
- *Activation Function (Final Layer):* Sigmoid (compared with Softmax, Sigmoid performed better)
- *Data Augmentation:* Experimented but found that training without augmentation gave better results.

### Training Process
- *Initial Weights Saved & Loaded:* Ensuring consistent training across different runs.
- *Validation Set Used:* To monitor generalization.
- *Training Accuracy & Loss Plotted.*

---

### Results & Performance
- *Test Accuracy:* *96.74%*
- *Final Model Saved As:* best_model_partb.pth

---

### Evaluation
- *The CNN model significantly outperformed ML classifiers.*
- *Test accuracy (96.74%)* indicates strong generalization.
- *Further improvements:* More hyperparameter tuning, deeper architectures, and additional data preprocessing.


---

## Conclusion
This project demonstrates the effectiveness of *CNNs for binary classification* in a face mask detection scenario. By tuning hyperparameters and optimizing model architecture, we achieved a high accuracy of *96.74%*, outperforming traditional ML classifiers.



### PART C

2 techniques used: K-means clustering based segmentation ,Otsu's Thresholding and Region-growing.

For K-means, k=2, one for mask region and another for backround.

Here for the choice of the 2 initial centroids, we use domain knowledge. The images are cropped to face-size which implies that it is higly likely that some region of the mask must be in the center of image. 

So we choose one centoid at center and another at corner.

<p align="center">

  <img src="Result_Images/000007_Ground_Truth.jpg" width="25%" />
<img src="Result_Images/000007_1.jpg" width="25%" />
  <img src="Result_Images/000007_1_kmeans.jpg" width="25%" />
</p>
[Results from K-means and ground truth mask for MSFD/1/000007.jpg]


<p align="center">

  <img src="Result_Images/000007_Ground_Truth.jpg" width="25%" />
<img src="Result_Images/000007_1.jpg" width="25%" />
  <img src="Result_Images/000007_1_Otsu.jpg" width="25%" />
</p>
[Results from Otsu and ground truth mask for MSFD/1/000007.jpg]
<!-- 
<p align="center">
    <img src="Part-C/MSFD/1/img/000007.jpg" width="25%" />
  <img src="Part-C/MSFD/1/face_crop_segmentation/000007_1.jpg" width="25%" />
  <img src="Part-C/MSFD/1/face_crop_segmentation_regiongrowing_pred/000007_1.jpg" width="25%" />
</p>
[Results from K-means and ground truth mask for MSFD/1/000007.jpg] -->



In conclusion:
| K-means | Otsu |
|--------|------------|
| Higher IOU | Lower IOU|
| Higher Dice Score | Lower Dice Score|

Both the algorihtms rely on predefined parameters, they do not 'learn' and hence fail to generalise over large dataset (poor mean IoU and Dice scores). Computing mean IoU and Dice for K-means over 8500+ images is computationally expensive, moreover it is evident from its performance over random samples that its scores won't be significantly better Otsu Thresholding.

---


### PART D


## i. Introduction
This project focuses on implementing image segmentation techniques using both traditional region-based methods and deep learning models such as CNN and U-Net. The objective is to segment facial regions accurately and compare the effectiveness of different methodologies.

## ii. Dataset
- *Source*: The dataset used consists of cropped facial images with corresponding ground truth masks.
- *Structure*:
  - face_crop/: Contains input images.
  - face_crop_segmented/: Contains ground truth segmentation masks.
  - output/: Stores results from segmentation techniques.

## iii. Methodology
### *Traditional Segmentation (Part C)*
- *Region-Based Segmentation*: Methods such as flooding and binary closing were applied.
- *K-Means Clustering*: Used to segment regions based on color similarity.

### *Deep Learning Models (Part D)*
- *CNN-based Segmentation*: Trained on facial images to predict masks.
- *U-Net Architecture*: A powerful fully convolutional network trained for pixel-wise classification.

## iv. Hyperparameters and Experiments
- *CNN Model*:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32
  - Number of Epochs: 30
  - Loss Function: Categorical Crossentropy

- *U-Net Model*:
  - Optimizer: Adam
  - Learning Rate: 0.0001
  - Batch Size: 16
  - Number of Epochs: 20
  - Loss Function: Dice Loss

    In this project, we experimented with different hyperparameters to optimize the performance of CNN and U-Net models for image segmentation. Below are the key hyperparameters used:

    1. Learning Rate
    Value Used: 0.0001
    
    Optimizer: Adam (torch.optim.Adam)
    
    Reasoning: A small learning rate ensures stable convergence and prevents overshooting the optimal weights.
    
    2. Batch Size
    Value Used: Defined as BATCH_SIZE (used in DataLoaders)
    
    Impact: Controls the number of samples processed before updating model weights, affecting training stability and speed.
    
    3. Number of Epochs
    Value Used: 30
    
    Training Strategy: The model is trained over 30 iterations to allow convergence without overfitting.
    
    4. Optimizer
    Type: Adam (torch.optim.Adam)
    
    Reasoning: Adam is widely used for deep learning tasks due to its adaptive learning rate properties.
    
    5. Loss Function
    Finding: The loss function was not explicitly found in the extracted code.
    
    Potential Options: Could be CrossEntropyLoss or Dice Loss for segmentation.
    
    
    6. Activation Functions
    Finding: ReLU (torch.nn.ReLU) is likely used in CNN layers.
    
   
    
    Experiments and Variations
    The learning rate, batch size, and number of epochs can be adjusted for further fine-tuning.
    
    Future experiments could explore different loss functions and data augmentation strategies to improve segmentation performance.
    
    
    Different variations of learning rates, optimizers, and batch sizes were tested to fine-tune the models.

## v. Results
- *Evaluation Metrics*:
  - Accuracy
  - Intersection over Union (IoU)
  - Dice Similarity Score

  <p align="center">
  <img src="images/unetpics.png" width="45%" />
  
</p>


| Model | Accuracy | IoU | Dice Score |
|--------|------------|------|------------|
| U-Net | 0.9664 | 0.9137 | 0.9509 |

As we can see the unet model works much better than traditional methods

<p align="center">
  <img src="images/unetresults" width="45%" />
  
</p>


## vi. Observations and Analysis
- *Traditional methods* work well for simple segmentation tasks but struggle with complex images.
- *CNN-based models* improve segmentation but may require extensive data augmentation.
- *U-Net* outperforms other approaches, providing the highest accuracy and IoU.
- Challenges include dealing with varying lighting conditions and occlusions, which were addressed using preprocessing techniques and data augmentation.

## 7. How to Run the Code
### Setup
1. Clone the repository:
   bash
   git clone https://github.com/DeathlyMade/VR-Project.git
   cd VR_PROJECT_1
   
2. Install dependencies:
    
   bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   
3. Download the dataet from the source specified and put the 2 repositores dataset and MSFD at same directory level, immediately below repository level. Make directory output, command mkdir output. Final structure must look like :
    
   ``` .
    ├── dataset
    ├── MSFD
    ├── output
    ├── scripts
    └── images

    ```
    
    # Other files like README.md, pdf, etc are not shown in this tree.
    
4. Run the scripts:
   
   \scripts contains 2 notebooks part_a_b.ipynb and part_c_d.ipynb, which contains scripts for the respective parts. They can be run all at once or one at a time to see partial results.

---

## 8. Conclusion
This project demonstrates the effectiveness of deep learning techniques for face mask detection and segmentation. CNN models outperform traditional classifiers for binary classification, while U-Net provides more precise segmentation results. Further improvements can be achieved by using more complex architectures and larger datasets.

---
