# Handwritten Digit Recognition with Pygame & TensorFlow

This project demonstrates **training a simple neural network on the MNIST dataset** using TensorFlow, and then performing **real-time inference** through an interactive **Pygame-based GUI** where you can draw digits and see predictions live.

## Features
- **Model Training**: Trains a fully connected neural network on MNIST handwritten digits.
- **Interactive Drawing Board**: Draw digits (0–9) directly on a Pygame canvas.
- **Live Prediction**: Displays predicted digit with confidence scores for all classes.
- **Softmax Output Visualization**: Highlights the most likely prediction in green.

## Requirements
Make sure you have the following installed:
```bash
pip install tensorflow pygame numpy
```

## Usage

### 1️⃣ Train the Model
Run the training script to create and save the neural network:
```bash
python training_using_TensorFlow.py
```
This will produce a file:
```
Model.h5
```

### 2️⃣ Run the Inference GUI
```bash
python inference_using_TensorFlow.py
```
**How it works:**
- A 28×28 grid simulates the MNIST input format.
- Draw using the left mouse button.
- Release the mouse to run prediction.
- Confidence scores appear on the right.
- Press **Enter** or close the window to exit.

## Model Architecture
- **Input Layer**: 784 neurons (flattened 28×28 image)
- **Hidden Layer 1**: 32 neurons, ReLU activation
- **Hidden Layer 2**: 16 neurons, ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0–9), linear activation + softmax during inference

## Demo
&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/user-attachments/assets/bc6d857f-165c-4740-95ac-b9e9e2f4c999" alt="Demo" width="500">
