#####################
# imports and useful start lines
#####################
import csv
import numpy as np
import matplotlib.pyplot as plt

#####################
# Algorithms
#####################
# Linear gradient descent
def compute_loss(y, tx, w):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.matmul(tx, w)
    te = np.transpose(e)
    N = len(y)
    return (np.matmul(te, e) / (2 * N))

def compute_loss_mae(y, tx, w):
    e = y - np.matmul(tx, w)
    N = len(y)
    return np.mean(np.abs(e))
def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return np.matmul(tx.transpose(), e) * (-1) / N

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: final optimized parameter w of shape(D, )
        loss: final loss (scalar) at max_iter with final optimized parameter w
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss

# Stochastic gradient descent for linear models
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w): 
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(batch_size=1, )
        tx: numpy array of shape=(batch_size=1, D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    
    batch_size = len(y)
    e = y - np.matmul(tx, w)
    return np.matmul(tx.transpose(), e) * (-1) / batch_size


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: final optimized parameter w of shape(D, ) (using SGD)
        loss: final loss (scalar) at max_iter with final optimized parameter w (using SGD)
    """

    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size = batch_size): # there is only one batch actually.
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss

# Least square regression using normal equations
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    XTX = tx.transpose() @ tx
    XTy = tx.transpose() @ y
    w = np.linalg.solve(XTX, XTy)
    mse = compute_loss(y, tx, w)
    return w, mse

# Ridge regression
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N = len(y)
    D = tx.shape[1]

    # Generating the diagnoal matrix lambda' * I (D * D) = Lambda_slash
    lambda_slash = 2 * lambda_ * N
    Lambda_slash = np.diag(lambda_slash * np.ones(D))

    Coef = (tx.transpose() @ tx) + Lambda_slash
    w_star = np.linalg.solve(Coef, tx.transpose() @ y)
    loss = compute_loss(y, tx, w_star)
    return w_star, loss


# Logistic regression 
# Comment: use this instead of e^t / (1 + e^t) for computational stability.
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w): # Note this is different from compute_loss from previous content
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
 
    N = y.shape[0]
    res = 0
    for n in range(0, N):
        xnt_w = tx[n].transpose() @ w
        res += np.log(1 + np.exp(xnt_w)) - y[n] * xnt_w
    res = res / N
    return np.squeeze(res)

def calculate_gradient(y, tx, w): # Note this is different from the compute_gradient in the previous content.
    N = y.shape[0]
    return (tx.transpose() @ (sigmoid(tx @ w) - y)) / N

def learning_by_gradient_descent(y, tx, w, gamma): # Do one step of gradient descent using logistic regression. 
    w_descent = w - gamma * calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w) # we are here calculating the loss corresponding the w before update.
    return loss, w_descent

def logistic_regression_gradient_descent(y, tx, initial_w, max_iter, gamma): # Do the iterative descent
    # init parameters
    losses = 0
    w = initial_w
    for iter in range(max_iter):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    return w, loss

# Regularized logistic regression
def penalized_logistic_regression(y, tx, w, lambda_): # Return the penalized loss and gradient
    loss_penalized = np.squeeze(calculate_loss(y, tx, w) + lambda_ * (w.transpose() @ w))
    gradient_penalized = calculate_gradient(y, tx, w) + (2 * lambda_ * w)
    return loss_penalized, gradient_penalized

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_): # Do one step of penalized logistic regression gd.
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w

def logistic_regression_penalized_gradient_descent(y, tx, lambda_, initial_w, max_iter, gamma): # Do the iterative descent
    w = initial_w
    loss = 0
    for iter in range(max_iter):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    return w, loss

# transformation： regression -> classification
def predict_labels(w, tx):
    y_pred = tx @ w
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

# Build polynomial features
def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    rows, cols = x.shape

    phi = np.zeros((rows, cols*(degree)))

    for i in range(rows):
        for j in range(cols):
            phi[i][j] = x[i][j]

    for d in range(2, degree+1):
        for i in range(rows):
            for j in range(cols):
                phi[i][j + (d-1)*cols] = x[i][j]**d

    return phi

#######################
# Writing the results
#######################
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline = '') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            if int(r2) == 0:
                r2 = -1
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


########################
# Loading the data
########################
def load_data(data_path):
    idx = np.genfromtxt(data_path, delimiter=",", skip_header = 1, usecols=[0])
    y = np.genfromtxt(data_path, delimiter=",", dtype=str, skip_header = 1, usecols=[1], converters={1:lambda x: 0 if x == b'b' else 1})
    x = np.genfromtxt(data_path, delimiter=",", skip_header = 1, usecols = list(range(2,32)))
    return y, x, idx

data_folder = "Data\\"
y, tx_train, idx_train = load_data(data_folder + "train.csv")
_, tx_test, idx_test = load_data(data_folder + "test.csv")



#################
# Cleaning the data
#################

# remove related features
def remove_related_features(tx):
    del_features = [5, 6, 12, 21, 24, 25, 26, 27, 28, 29]
    tx = np.delete(tx, del_features, 1)
    return tx

tx_train = remove_related_features(tx_train)
tx_test = remove_related_features(tx_test)

# spliting the data
def groupby_jetnum(tx): # Group by input data (related features removed) based on pri_jet_num
    group0 = (tx[:, 18] == 0)
    group1 = (tx[:, 18] == 1)
    group2 = (tx[:, 18] != 0) & (tx[:, 18] != 1)
    # for index 18, see imp_draft.ipynb
    return [group0, group1, group2]

groups_train = groupby_jetnum(tx_train)
groups_test = groupby_jetnum(tx_test)

# replacing na values with column mean
def replace_na_values(data):
    for i in range(data.shape[1]):
        msk = (data[:, i] != -999.)
        # Replace NA values with mean value
        median = np.median(data[msk, i])
        if np.isnan(median):
            median = 0
        data[~msk, i] = median
    return data

#tx_train = replace_na_values(tx_train)
#tx_test = replace_na_values(tx_test)
# It's better to deal with NAs within of subgroups below.


################
# Training model
################
degree = 4
lambda_ = 1e-5
preds = np.zeros(tx_test.shape[0])
for i in range(len(groups_train)):
    feature_tr = tx_train[groups_train[i]]
    label_tr = y[groups_train[i]]
    feature_te = tx_test[groups_test[i]]
    
    feature_tr = replace_na_values(feature_tr)
    feature_te = replace_na_values(feature_te)
    
    phi_tr = build_poly(feature_tr, degree)
    phi_te = build_poly(feature_te, degree)
        
    w_star, _ = ridge_regression(label_tr, phi_tr, lambda_) #<---ridge regression
    
    #w_star, _ = least_squares(label_tr, phi_tr) <--------- least squares
    
    #w_init = np.zeros(feature_tr.shape[1])
    #w_star, _ = least_squares_GD(label_tr, feature_tr, w_init, 50, 10e-10) <------------- gradient descent
    
    #w_init = np.zeros(feature_tr.shape[1])
    #w_star, _ = logistic_regression_penalized_gradient_descent(label_tr, feature_tr, 1e-10, w_init, 50, 1e-10) <---regularized logistic
    
    #w_init = np.zeros(feature_tr.shape[1])
    #w_star, _ = logistic_regression_gradient_descent(label_tr, feature_tr, w_init, 50, 1e-10) <--------- logistic regression

    pred_y = predict_labels(w_star, phi_te) #<---all but logistic
    
    #pred_y = sigmoid(feature_te @ w_star)
    #pred_y[np.where(pred_y <= 0.5)] = 0
    #pred_y[np.where(pred_y > 0.5)] = 1  #<---(regularized) logistic
    
    preds[groups_test[i]] = pred_y 
    
# Generate file csv for submission
OUTPUT_PATH = data_folder + 'submission.csv'
create_csv_submission(idx_test, preds, OUTPUT_PATH)
print('Submission file created!')



# Ridge regression : 0.807
# Logistic regression regularized (no poly): 0.694
# Least Squares (poly): 0.669
# Gradient descent (no ploy): 0.658
# Logistics regression: 0.694

#after removing relating columns:
# Ridge regression: 0.801
# Logistic regression regularized (ploy): 0.586 (wtf??)

