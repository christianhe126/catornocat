import numpy as np
import h5py
from PIL import Image

# Function to load the data
def load_data(train_file_path, test_file_path):
    with h5py.File(train_file_path, "r") as file:
        train_set_x_orig = np.array(file["train_set_x"][:])
        train_set_y_orig = np.array(file["train_set_y"][:])

    with h5py.File(test_file_path, "r") as file:
        test_set_x_orig = np.array(file["test_set_x"][:])
        test_set_y_orig = np.array(file["test_set_y"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ReLU function
def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU function
def relu_derivative(z):
    return z > 0

# Initialize parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# Compute cost
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))) / m
    cost = np.squeeze(cost)
    return cost

# Backward propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(cache['Z1'])
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1'] - learning_rate * grads['dW1']
    b1 = parameters['b1'] - learning_rate * grads['db1']
    W2 = parameters['W2'] - learning_rate * grads['dW2']
    b2 = parameters['b2'] - learning_rate * grads['db2']

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Model
def model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.01)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

# Prediction
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    return predictions

# Load the data
train_x_orig, train_y, test_x_orig, test_y = load_data('input/train_catvnoncat.h5', 'input/test_catvnoncat.h5')

# Reshape and standardize the data
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.

# Train the model
n_h = 4  # number of neurons in the hidden layer
parameters = model(train_x_flatten, train_y, n_h, num_iterations=5000, print_cost=True)

# Predict on test data
predictions = predict(parameters, test_x_flatten)
correct_predictions = np.dot(test_y, predictions.T) + np.dot(1-test_y, 1-predictions.T)
accuracy = (np.squeeze(correct_predictions) / test_y.size) * 100
print('Test Accuracy: {:.2f}%'.format(accuracy))

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img.reshape((1, 64*64*3)).T
    img = img / 255.
    return img

def predict_shallow(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    return predictions

#### Case 1:
# Load and preprocess your image
image_path = 'images/cat1.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
predictions = predict_shallow(parameters, preprocessed_image)
print("This image1 is a", "cat" if predictions[0,0] == 1 else "non-cat")

#### Case 2:
# Load and preprocess your image
image_path = 'images/cat2.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
predictions = predict_shallow(parameters, preprocessed_image)
print("This image1 is a", "cat" if predictions[0,0] == 1 else "non-cat")

#### Case 3:
# Load and preprocess your image
image_path = 'images/cat3.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
predictions = predict_shallow(parameters, preprocessed_image)
print("This image1 is a", "cat" if predictions[0,0] == 1 else "non-cat")

#### Case 4:
# Load and preprocess your image
image_path = 'images/cat4.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
predictions = predict_shallow(parameters, preprocessed_image)
print("This image1 is a", "cat" if predictions[0,0] == 1 else "non-cat")