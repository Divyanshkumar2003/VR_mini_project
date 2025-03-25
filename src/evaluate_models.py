import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import tensorflow as tf

# Load Test Data
X_test = np.load("../dataset/X_faces.npy")
Y_test = np.load("../dataset/Y_masks.npy")

X_test_resized = np.array([cv2.resize(img, (224, 224)) for img in X_test])

# Load Models
cnn_model1 = tf.keras.models.load_model("../models/mask_detector_adam_relu.keras")
cnn_model2 = tf.keras.models.load_model("../models/mask_detector_adam_tanh.keras")
svm_model = joblib.load("../models/svm_model.pkl")
mlp_model = joblib.load("../models/mlp_model.pkl")
# Evaluate CNN
cnn1_pred = (cnn_model1.predict(X_test_resized) > 0.5).astype(int)
cnn1_acc = accuracy_score(Y_test, cnn1_pred)


cnn2_pred = (cnn_model2.predict(X_test_resized) > 0.5).astype(int)
cnn2_acc = accuracy_score(Y_test, cnn2_pred)

# Evaluate SVM
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(Y_test, svm_pred)

print(f"CNN Accuracy: {cnn_acc}")
print(f"SVM Accuracy: {svm_acc}")

