VR MINI PROJECT
Mask Classification and Segmentation Project

Introduction

This project focuses on classifying face images based on the presence of a mask and performing segmentation to identify mask regions. The classification is performed using both traditional machine learning approaches with handcrafted features and deep learning approaches using Convolutional Neural Networks (CNN). Additionally, region-based segmentation is applied using both traditional image processing techniques and U-Net for precise segmentation.

Directory Structure-:

Mask_Classification_Project/
    ├── dataset/
    │   ├── without_mask/
    │   ├── with_mask/
    ├── models/
    │   ├── svm_model
    │   ├── mlp_model
    │   ├── cnn_model
    │   ├── unet_model
    ├── scripts/
    │   ├── train_ml_classifiers.py
    │   ├── train_cnn.py
    │   ├── train_unet.py
    │   ├── segment_traditional.py
    ├── env/
    ├── requirements.txt
    ├── README.md

Dataset-

The dataset contains images of faces categorized into two classes:

1.Without Mask

2.With Mask

The dataset is stored in a directory structure as follows:
../dataset/
    ├── without_mask/
    ├── with_mask/
    
Methodology

A. Binary Classification Using Handcrafted Features and ML Classifiers

1.Feature Extraction

Handcrafted features are extracted using Histogram of Oriented Gradients (HOG) to capture texture information from face images. These features serve as input for the classifiers.

2.Model Training

Two machine learning models are trained:

i).Support Vector Machine (SVM) - Uses a linear kernel to separate classes based on HOG features.

ii).Multilayer Perceptron (MLP) Classifier - A neural network with two hidden layers (128 and 64 neurons) trained using backpropagation.

Part B: CNN-Based Binary Classification-:

1. Dataset Preparation
Loads images from ../dataset/, normalizes pixel values, and splits into training (80%) and validation (20%) sets.

Uses ImageDataGenerator to preprocess images (resizing to 224x224).

2. Model Design (Transfer Learning with MobileNetV2)
Uses MobileNetV2 (pre-trained on ImageNet) as the feature extractor.

Adds a Flatten layer, a Dense layer (128 neurons, ReLU/Tanh activation), and an output layer (Sigmoid activation for binary classification).

3. Training and Hyperparameter Comparison
Optimizer: Tests Adam (adaptive learning rate) and SGD (momentum-based gradient descent).

Activation Function: Compares ReLU (fast convergence) vs. Tanh (smooth gradient flow).

Loss Function: Uses Binary Cross-Entropy since it’s a classification task.

Trains for 10 epochs and evaluates accuracy.

4. Model Saving
Saves each trained model (mask_detector_optimizer_activation.keras) in ../models/ for future use.

C. Region Segmentation Using Traditional Techniques-:
To segment the mask region in "with mask" images, the following traditional methods are applied:

1.Gaussian Blur: Smooths the image by reducing noise and details, making thresholding and edge detection more effective. This helps in minimizing false edges caused by noise.

2.Thresholding: Converts the blurred grayscale image into a binary format by setting a pixel intensity threshold. This separates the mask from the background but may struggle with lighting variations.

3.Edge Detection (Canny): Identifies the boundaries of the mask by detecting sharp intensity changes. It enhances mask contours but may require fine-tuning to avoid detecting unnecessary edges.

These techniques provide a basic segmentation approach, though they may lack the precision of deep-learning-based methods.

D. Mask Segmentation Using U-Net

Model Design

A U-Net model is designed for precise mask segmentation. It follows an encoder-decoder structure with skip connections to retain spatial information.

1.Dataset Preparation

The dataset consists of pairs of images:

Input images (128x128 face images)

Mask ground truth (128x128 binary masks indicating mask regions)

Both datasets are normalized before training.

2.Model Training

The U-Net model is trained using binary cross-entropy loss and optimized using Adam optimizer.

3.Evaluation

The segmentation performance is measured using:

i)Intersection over Union (IoU)

ii)Dice Score

The performance is compared with traditional segmentation techniques.

Models Performance Summary
Our project evaluates different machine learning models for face mask detection. The table below summarizes the accuracy of CNN (with different activation functions), SVM, and MLP classifiers.

Model    Accuracy
CNN (Adam + ReLU)    97.40%
CNN (Adam + Tanh)    2.60%
SVM    33.20%
MLP    28.80%
📌 Key Observations
CNN with Adam + ReLU achieved the highest accuracy (97.40%), making it the most reliable choice.

CNN with Adam + Tanh performed poorly (2.60%), indicating that Tanh is not suitable for this binary classification task.

SVM and MLP models underperformed (33.2% and 28.8%), suggesting that handcrafted HOG features may not be as effective as deep learning-based feature extraction.

Note-
Link for the model files is given below as we were unable to upload on github because of the file size exceeding the limit-:
https://iiitbac-my.sharepoint.com/personal/sparsh_salodkar_iiitb_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsparsh%5Fsalodkar%5Fiiitb%5Fac%5Fin%2FDocuments%2Fmodels&ga=1

📌 Conclusion
From our evaluation, CNN (Adam + ReLU) is the best-performing model for face mask classification, significantly outperforming traditional ML models like SVM and MLP.

Observations and Analysis-:

A. Binary Classification Using Handcrafted Features and ML Classifiers

Insights: HOG features effectively capture texture, and SVM performs well for small datasets. MLP achieves better accuracy but requires tuning.

Challenges: Feature extraction is time-consuming, and models struggle with complex patterns.

Solutions: Optimizing hyperparameters and increasing dataset diversity improve performance.

B. Binary Classification Using CNN

Insights: CNNs automatically learn hierarchical features, achieving higher accuracy than ML models.

Challenges: Training requires large datasets and computational resources.

Solutions: Data augmentation and transfer learning improve model generalization.

C. Region Segmentation Using Traditional Techniques

Insights: Gaussian Blur helps smooth noise, and Canny edge detection highlights mask boundaries.

Challenges: Variations in lighting and facial structure reduce segmentation accuracy.

Solutions: Adaptive thresholding and fine-tuned edge detection improve robustness.

D. Mask Segmentation Using U-Net

Insights: U-Net provides precise segmentation due to its encoder-decoder structure and skip connections.

Challenges: Training requires high-quality ground truth masks and large datasets.

Solutions: Increasing training data and using data augmentation enhance segmentation accuracy.


How to Run the Code-:

Installation

Ensure you have all dependencies installed by running-:
pip install -r requirements.txt

Virtual Environment Setup -:
python3 -m venv env
source env/bin/activate



Authors-
Sparsh Salodkar IMT2022113
Divyansh Kumar  IMT2022509
Raghav Khurana  IMT2022550

