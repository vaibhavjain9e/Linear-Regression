import numpy as np


def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    err = np.mean((np.subtract(np.dot(X, w), y))**2)
    return err


def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
    return w


def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    X_transpose_X = np.dot(np.transpose(X), X)
    smallest_eigen_value = 0
    while smallest_eigen_value < 1e-5:
        smallest_eigen_value = np.min((np.linalg.eig(X_transpose_X)[0]))
        if smallest_eigen_value < 1e-5:
            X_transpose_X = X_transpose_X + 1e-1*np.identity(X_transpose_X.shape[0])

    w = np.dot(np.dot(np.linalg.inv(X_transpose_X), np.transpose(X)), y)
    return w


def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    X_transpose_X = np.dot(np.transpose(X), X)
    X_transpose_X = X_transpose_X + lambd * np.identity(X_transpose_X.shape[0])
    w = np.dot(np.dot(np.linalg.inv(X_transpose_X), np.transpose(X)), y)
    return w


def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    bestlambda = 0
    err = 1

    for int_value in range(-19, +20):
        if int_value >= 0:
            float_value = float("1e+" + str(int_value))
        else:
            float_value =float("1e" + str(int_value))
        w = regularized_linear_regression(Xtrain, ytrain, float_value)

        mean_error = mean_square_error(w, Xval, yval)
        if err > mean_error:
            err = mean_error
            bestlambda = float_value

    return bestlambda
    

def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    X_temp = X[:]
    for i in range(2, power + 1):
        x = np.power(X, i)
        for col in range(0, x.shape[1]):
            X_temp = np.insert(X_temp, X_temp.shape[1], x[:, col], axis=1)

    return X_temp


