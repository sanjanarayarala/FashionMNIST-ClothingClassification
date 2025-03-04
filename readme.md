## **README: Fashion MNIST Clothing Classification**

### **Project Overview**
This project is a **Fashion MNIST classification model** that predicts clothing categories using a deep neural network (DNN) built with **TensorFlow & Keras**. The dataset consists of **60,000 training images** and **10,000 test images**, each being a **28x28 grayscale image** representing one of **10 fashion categories**.

The model uses **fully connected layers (Dense layers)** with **ReLU activation** and is optimized using different strategies, including **SGD, Adam, and RMSprop**. Additionally, **Keras Tuner** is used for **hyperparameter optimization**.

---

### **Dataset Information**
The **Fashion MNIST dataset** contains:
- **60,000 training images**
- **10,000 test images**
- **10 clothing categories:**

  1.T-shirt/top  
  2.Trouser  
  3.Pullover  
  4.Dress  
  5.Coat  
  6.Sandal  
  7.Shirt  
  8.Sneaker  
  9.Bag  
  10.Ankle boot  

Each image is **28x28 pixels** and has pixel intensity values between **0 and 255**.

---

### **Project Structure**
```
FashionMNIST-Classification/
│-- ClothingClassification.ipynb    # Jupyter Notebook with the full implementation
│-- fashion_mnist_model.png         # Model architecture visualization
│-- kt_fashion_mnist/               # Directory for Keras Tuner search results
│-- README.md                        # This file
```

---

### **Implementation Steps**
#### **1. Load and Preprocess the Data**
- Download the **Fashion MNIST** dataset from `keras.datasets`.
- Normalize pixel values by **dividing by 255** (feature scaling).
- Split the training set:  
  - **First 5,000** images → **Validation set**  
  - **Remaining 55,000** images → **Training set**  

#### **2. Visualize the Dataset**
- Display sample images using `matplotlib` to understand the dataset structure.
- Assign **class names** for better readability instead of numerical labels.

#### **3. Build the Neural Network**
- **Input layer**: Flattens the **28x28** image into a **1D array** (784 values).
- **Hidden layers**:
  - **Layer 1**: 300 neurons, ReLU activation
  - **Layer 2**: 100 neurons, ReLU activation
- **Output layer**: 10 neurons (one for each class) with **Softmax activation**.
  
#### **4. Train the Model**
- **Optimizer choices**:  
  - **SGD** (Stochastic Gradient Descent)  
  - **Adam**  
  - **RMSprop**  
- **Loss function**: Sparse Categorical Crossentropy  
- **Metrics**: Accuracy  
- Train for **15 epochs** with validation data.

#### **5. Evaluate Model Performance**
- **Plot training history**: Visualize loss and accuracy over epochs.
- **Test the model**: Evaluate performance on **test set**.
- **Make Predictions**: Predict classes for sample images.

#### **6. Hyperparameter Optimization (Using Keras Tuner)**
- Tune the model’s hyperparameters using **Keras Tuner**:
  - **Number of neurons in hidden layers** (range: **128–512**, step: **64**)
  - **Activation function** (`ReLU` or `Tanh`)
  - **Optimizer** (`Adam`, `SGD`, `RMSprop`)
- **Search for best hyperparameters** using `RandomSearch`.
- **Train the best model** using optimal parameters.

#### 🔧 Model Training & Hyperparameter Tuning
Optimization Techniques Used                                                                                       
Adam optimizer (selected through hyperparameter tuning)                                                            
Hyperparameter tuning with KerasTuner to find the best architecture.                                               
Early stopping to avoid overfitting.                                                                               
Hyperparameter Tuning Results                                                                                      
After 10 trials, the best model configuration found:                                                               
                                                                                                                
Hidden Layer 1: 320 neurons, activation = ReLU                                                                     
Hidden Layer 2: 128 neurons, activation = ReLU                                                                     
Optimizer: Adam                                                                                                    

#### 📊 Results
Final Model Training Performance                                                                                   
Metric	Value                                                                                                      
- Training Accuracy	93.24%
- Validation Accuracy	89.20%
- Test Accuracy	88.13%
- Test Loss	0.4232
Training Progress                                                                                                  
- Epoch 1: Accuracy 78.42%, Validation Accuracy 86.60%
- Epoch 5: Accuracy 89.47%, Validation Accuracy 88.94%
- Epoch 10: Accuracy 91.70%, Validation Accuracy 89.38%
- Epoch 15: Accuracy 93.24%, Validation Accuracy 89.20%                                                            
The model achieved a final test accuracy of 88.13%, indicating strong generalization on unseen data.

### **Requirements**
To run this project, install the necessary dependencies:

```bash
pip install tensorflow keras matplotlib numpy pandas keras-tuner
```

Or install them using `requirements.txt` (if provided):

```bash
pip install -r requirements.txt
```

---

### **How to Run**
1. Clone the repository or download the Jupyter Notebook.
2. Open **ClothingClassification.ipynb** in Jupyter Notebook.
3. Run the notebook cells sequentially.

---

### **Future Improvements**
- Implement **CNN (Convolutional Neural Network)** for better accuracy.
- Experiment with **data augmentation** to improve generalization.
- Try **transfer learning** with a pre-trained model like **MobileNet** or **ResNet**.

---


