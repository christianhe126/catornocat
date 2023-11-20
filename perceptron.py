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

# Load the data
train_x_orig, train_y, test_x_orig, test_y = load_data('input/train_catvnoncat.h5', 'input/test_catvnoncat.h5')

# Reshape and standardize the data
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

# Propagate function
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    
    return grads, cost

# Optimization function
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

# Predict function
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    return Y_prediction

# Model function
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test, 
        "Y_prediction_train" : Y_prediction_train, 
        "w" : w, 
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations
    }
    
    return d

# Running the model
d = model(train_x_flatten, train_y, test_x_flatten, test_y, num_iterations=230, learning_rate=0.005, print_cost=True)

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img.reshape((1, 64*64*3)).T
    img = img / 255.
    return img

#### Case 1:
# Load and preprocess your image
image_path = 'images/cat1.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
w = d['w']
b = d['b']
prediction = predict(w, b, preprocessed_image)

print("This image1 is a", "cat" if prediction[0,0] == 1 else "non-cat")

#### Case 2:
# Load and preprocess your image
image_path = 'images/cat2.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
w = d['w']
b = d['b']
prediction = predict(w, b, preprocessed_image)

print("This image2 is a", "cat" if prediction[0,0] == 1 else "non-cat")

#### Case 3:
# Load and preprocess your image
image_path = 'images/cat3.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
w = d['w']
b = d['b']
prediction = predict(w, b, preprocessed_image)

print("This image3 is a", "cat" if prediction[0,0] == 1 else "non-cat")

#### Case 4:
# Load and preprocess your image
image_path = 'images/cat4.jpg'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the model
w = d['w']
b = d['b']
prediction = predict(w, b, preprocessed_image)

print("This image4 is a", "cat" if prediction[0,0] == 1 else "non-cat")