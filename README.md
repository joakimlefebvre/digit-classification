# MNIST Digit Classification with Convolutional Neural Network (CNN)
This repository contains a simple implementation of a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The model is built using TensorFlow/Keras and includes key elements such as data preprocessing, model architecture, training, evaluation, and visualization of results.

## 0. Requirements

- Python 3.x
- TensorFlow >= 2.0
- Matplotlib
- NumPy
- Seaborn
- Scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow matplotlib numpy seaborn scikit-learn
```
## 1. Dataset
The MNIST dataset is a collection of 60,000 28x28 grayscale images of handwritten digits (0-9) for training, and 10,000 images for testing. The dataset is included with Keras and will be automatically loaded during runtime.
![image](https://github.com/user-attachments/assets/41a96f9c-a406-4682-8658-cac08d229880)
## 2. Load and Preprocess Data
### 2.1 Loading the Dataset
```python
keras.datasets.mnist.load_data()
```
  - `x_train`,`y_train`: 60000 training images and labels.
  - `x_test`,`y_test`: 10000 test images and labels.
### 2.2 Data Preprocessing
- Reshaping the image data to `(28,28,1)`to be compatible with the CNN input, 1 color channel (grayscale).
- Normalization: Pixel values range from 0 (black) to 255 (white). We scale them to the range [0, 1] by dividing all pixel values by 255.
- One-Hot Encoding: The target labels are one-hot encoded using `tf.keras.utils.to_categorical()`
#### One-Hot Encoding

In the MNIST dataset, the target labels represent the digits from `0` to `9`. Machine learning models, especially neural networks, require numerical representations of categorical data to perform computations. One way to represent the categorical labels is through **one-hot encoding**.

#### What is One-Hot Encoding?

**One-hot encoding** is a method to convert categorical labels into a binary matrix, where each class label is represented as a vector with all zeros except for a 1 in the position corresponding to that class label. 

For example, in the case of digit classification (0-9), the label `7` would be represented as a 10-element vector, where only the 8th position (corresponding to digit 7) is `1` and all other positions are `0`. This prevents the model from assuming any inherent order between the digits.

#### One-Hot Encoding for MNIST Labels

Let’s take a look at how the labels are one-hot encoded for the MNIST dataset:

- **Before One-Hot Encoding**: Each label is a single integer from `0` to `9` representing the corresponding digit in the image.
  
  | Label |
  |-------|
  | 7     |
  | 2     |
  | 3     |
  | 8     |

- **After One-Hot Encoding**: Each label is represented as a 10-element vector (since there are 10 classes: 0-9), where only the index corresponding to the label is `1` and all other positions are `0`.

  | Label | One-Hot Encoded Vector       |
  |-------|-----------------------------|
  | 7     | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] |
  | 2     | [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] |
  | 3     | [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] |
  | 8     | [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] |
## 3. Model Architecture
The CNN model consists of the following layers:\
- **Input Layer**: Takes input of shape `(28,28,1)`(grayscale image).
- **Convolutional Layers**:
  - First Layer: 32 filters, 3x3 kernel size, ReLU activation.
  - Second Layer: 64 filters, 3x3 kernel size, ReLU activation.
  - Third Layer: 64 filters, 3x3 kernel size, ReLU activation.
- **Max-Pooling Layers**: 2x2 pool size, applied after each convolutional layer.
- **Dropout Layers**: Applied after each convolutional layer to prevent overfitting.
- **Fully Connected (Dense) Layer**: 64 units, ReLU activation.
- **Output Layer**: 10 units (one for each class), softmax activation to output class probabilities. Softmax formula:\
$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$
<p align="center">
<img width="643" alt="image" src="https://github.com/user-attachments/assets/7c55ae3c-0124-484b-b496-4a3a71f11dc7" />
</p>

## 4. Model Training

The model is compiled with the following settings:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy (since this is a multi-class classification problem with a one-hot representation)
- **Metrics**: Accuracy

The model is trained using `model.fit()` with:

- **Batch Size**: 128  
- **Epochs**: 10  

---

## 5. Model Evaluation

The model's performance is evaluated on the test set using `model.evaluate()`.  

Key evaluation metrics:

- **Accuracy & Loss**:  
  - Training and validation accuracy/loss are plotted to visualize model performance over epochs.
<p align="center>
<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="https://github.com/user-attachments/assets/6aae639a-89fe-46f2-a007-5a914502bd75" width="400"/>
    <img src="https://github.com/user-attachments/assets/f4253e49-2f86-4d8e-8b01-03a55933a375" width="400"/>
</div>
</p>
  
### ⚠️ Warning: Use of Test Data as Validation

In this project, the **test dataset** is also used as the **validation dataset**. This decision was made because **MNIST is a relatively easy problem**, and the model is not being fine-tuned extensively.  

However, in a real-world scenario, it is **best practice** to:
- **Use a separate validation set** (e.g., split part of the training data for validation).
- **Reserve the test set exclusively for final evaluation** to prevent data leakage and ensure unbiased performance metrics.

For optimal training, you can create a validation split like this:

```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

history = model.fit(x_train, y_train, 
                    batch_size=128, 
                    epochs=10, 
                    validation_data=(x_val, y_val))
```
---

## 6. Predictions and Visualizations

The trained model makes predictions on the test dataset.  

### **Displaying Test Images**
- **30 test images** are displayed along with their predicted labels.  
- Correct predictions are shown in **black**.  
- Incorrect predictions are shown in **red**, with the correct label in parentheses.  

### **Confusion Matrix**
- A **heatmap** of the confusion matrix is generated using **Seaborn**.  
- This helps visualize how well the model predicts each digit class (0-9).
## 7. Model Saving
After training and evaluation, the model is saved to a `.keras` file.
```python
model.save(f'{model_name}.keras')
```
And can be loaded with:
```python
model = tf.keras.models.load_model(f'{model_name}.keras')
```
