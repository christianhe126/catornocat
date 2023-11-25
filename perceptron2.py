import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression

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
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1) / 255.
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1) / 255.

# Reshape the labels to be compatible with scikit-learn
train_y = train_y.reshape(-1)
test_y = test_y.reshape(-1)

# Create and train the logistic regression model
clf = LogisticRegression(random_state=0, penalty='l2', max_iter=10000)
clf.fit(train_x_flatten, train_y)

# Predictions
train_predictions = clf.predict(train_x_flatten)
test_predictions = clf.predict(test_x_flatten)

# Calculate accuracies
train_accuracy = np.mean(train_predictions == train_y) * 100
test_accuracy = np.mean(test_predictions == test_y) * 100

print(f"Train accuracy: {train_accuracy}%")
print(f"Test accuracy: {test_accuracy}%")
