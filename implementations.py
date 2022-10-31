""" Module containing all implementations of ML techniques required for the project """

import numpy as np
import csv
import copy
import random


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_loss(y, x, w, mae=False):

    e = y - x @ w
    if mae:
        loss = np.mean(np.abs(e))
    else:
        loss = np.mean(e ** 2) / 2
    return loss

def compute_subgradient_mae(y, x, w):
    n = x.shape[0]
    e = y - x @ w
    grd = -(x.T @ np.sign(e)) / n
    return grd

def compute_gradient_regress(y, x, w):
    
    pred = sigmoid(x.dot(w))
    grd = x.T @ (pred - y)
    
    return grd

def sigmoid(x):
    
    return 1.0 / (1 + np.exp(-x))

def compute_loss_regress(y, x, w):
    pred = sigmoid(x.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss)

def penalized_logistic_regression(y_tr, tx, w, lambda_):
    num_samples = y_tr.shape[0]
    loss = compute_loss_regress(y_tr, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = compute_gradient_regress(y_tr, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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

    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def compute_mse_loss(y, tx, w):

    e = y - tx.dot(w)
    mse_loss=1/2*np.mean(e**2)
    return mse_loss
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return  w,loss

def least_squares_SGD(y, tx,initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the last weight
        loss: the last loss
       
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    loss=None
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, e = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return w,loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: the last weight
        loss: the last loss

    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    return w,loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: the last weight
        loss: the last loss
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    return w,loss

def logistic_regression(y, tx,initial_w, max_iters, gamma):
    """implement logistic regression.
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the last weight
        loss: the last loss
       
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_regress(y, tx, w)
        loss = compute_loss_regress(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
    return w,loss

def reg_logistic_regression(y, tx,lambda_,initial_w, max_iters, gamma):
    """implement regularized logistic regression.
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        lambda_: scalar
        initial_w: The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the stepsize
        
    Returns:
        w: the last weight
        loss: the last loss
       
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss,w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # update w by gradient descent
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
    return w,loss




####### funtions for this project #######

def binarize(y, target_low=-1, target_high=1, threshold=0):
    y[y <= threshold] = target_low
    y[y > threshold] = target_high
    return y


def predict_labels(weights, data):

    y_pred = np.dot(data, weights)
    return binarize(y_pred)

def compute_accuracy(predict, targets):

    return np.mean(predict == targets)


def map_target_classes_to_bool(y):

    return (y == 1).astype(int)


def create_csv_submission(ids, y_pred, name):

    with open(name, 'w',newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})





def sigmoid(x):
    
    return 1.0 / (1 + np.exp(-x))


def compute_loss_reg_regress(y, x, w, lambda_=0):
    
    def safe_log(x, MIN=1e-9):
        """
        Return the stable floating log (in case where x was very small)
        """
        return np.log(np.maximum(x, MIN))

    predict = sigmoid(x @ w)
    log_pos, log_neg = safe_log(predict), safe_log(1 - predict)
    loss = -(y.T @ log_pos + (1 - y).T @ log_neg)
    loss += lambda_ * w.dot(w).squeeze()
    return loss


def compute_gradient_reg_regress(y, x, w, lambda_=0):
    
    predict = sigmoid(x @ w)
    grd = x.T @ (predict - y)
    grd += 2 * lambda_ * w
    return grd


def compute_hessian_reg_regress(y, x, w, lambda_=0):

    sgm = sigmoid(x @ w)
    #print("sgm:",sgm)
    s = sgm * (1 - sgm) + 2 * lambda_
    #print("s:",s)
    return (x.T * s) @ x


def compute_loss_hinge(y, x, w, lambda_=0):


    return np.clip(1 - y * (x @ w), 0, None).sum() + (lambda_ / 2) * w.dot(w)


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    return  input_data, yb, ids


def z_normalize_data(x, mean_x=None, std_x=None):
    
    if mean_x is None or std_x is None:
        ## ignore nan value
        mean_x, std_x = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    if np.any(std_x == 0):
        print(x[:, std_x == 0])
    x_norm = (x - mean_x) / std_x
    return x_norm, mean_x, std_x



def split_data_by_categories( x,y, ids, PRI_JET_NUM_index):


    ## divide the data into 3 subsets
    ## This implementation is too intrinsic! Check out 0.807_haolli/implementations.py: groupby_jetnum()
    category_list =[]
    """"""
    category_list.append(np.where(x[:, PRI_JET_NUM_index] == 0)[0])  #category 0
    category_list.append(np.where(x[:, PRI_JET_NUM_index] == 1)[0])  #category 1 
    category_list.append(np.where(np.logical_or(x[:, PRI_JET_NUM_index] == 2,   x[:, PRI_JET_NUM_index] == 3)) [0]) #category 2 
    
    ### delete the PRI_JET_NUM column
    x_split = [np.delete(x[indices, :], PRI_JET_NUM_index, axis=1) for indices in category_list]
    y_split = [y[indices] for indices in category_list]
    ids_split = [ids[indices] for indices in category_list]

    return x_split, y_split, ids_split


def remove_nan_columns(x):
    # Remove columns that are all filled with NA or 0 or (NA and 0)
    # It is still possible that we have columns with some NA and 0 values
    na_mask = np.isnan(x)
    zero_mask = x == 0
    na_columns = np.all(na_mask | zero_mask, axis=0)
    return x[:, ~na_columns]



def remove_correlated_features(x, min_abs_correlation):


    variances = np.nanvar(x, axis=0)
    correlation_coefficients = np.ma.corrcoef(np.ma.masked_invalid(x), rowvar=False)
    rows, cols = np.where(np.abs(correlation_coefficients) > min_abs_correlation)
    columns_to_remove = []
    for i, j in zip(rows, cols):
        if i >= j:
            continue
        if variances[i] < variances[j] and i not in columns_to_remove:
            columns_to_remove.append(i)
        elif variances[j] < variances[i] and j not in columns_to_remove:
            columns_to_remove.append(j)
    return np.delete(x, columns_to_remove, axis=1), columns_to_remove

""""""

def build_poly_and_cross(x, degree=2, cross_term=True):
    
    n, d = x.shape
    powers = [x ** deg for deg in range(1, degree + 1)]
    phi = np.concatenate((np.ones((n, 1)), *powers), axis=1)
    if cross_term:
        new_feat = np.array([x[:, i] * x[:, j] for i in range(d) for j in range(i + 1, d)]).T
        phi = np.append(phi, new_feat, axis=1)
    return phi


def preprocess_data(data, nan_value=-999., low_var_threshold=0.1, corr_threshold=0.9,
                           degree=2, cross_term=True, columns_to_remove=None, norm_first=True, mean=None, std=None):

    data = data.copy()
    data[data == nan_value] = np.nan
    data = remove_nan_columns(data)
    
    """"""
    if columns_to_remove is not None: ## for test dataset
        data = np.delete(data, columns_to_remove, axis=1)
    else:   ## for train dataset
        data, columns_to_remove = remove_correlated_features(data, corr_threshold)

    data = build_poly_and_cross(data, degree, cross_term)
    
    ## normalize
    data[:, 1:], mean, std = z_normalize_data(data[:, 1:], mean, std)
    data[np.isnan(data)] = 0.
    return data, columns_to_remove, mean, std



def reg_logistic_regression_with_val(y, x_tr,y_val, x_val, lambda_, initial_w, max_iters, gamma, threshold=1e-2):

    # Map classes from {-1, 1} to {0, 1}
    y = map_target_classes_to_bool(y)

    w = initial_w

    best_acc=-1
    best_w=None
    acc_log=[]
    try :   
        for n_iter in range(max_iters):
            # Compute the gradient and Hessian of the loss function
            grd = compute_gradient_reg_regress(y, x_tr, w, lambda_)
            hess = compute_hessian_reg_regress(y, x_tr, w, lambda_)

            # Update the weights
            
            w -= gamma / np.sqrt(n_iter+1) * np.linalg.solve(hess, grd)
            
            # Compute the current loss and test convergence
            loss = compute_loss_reg_regress(y, x_tr, w, lambda_)

            val_acc=compute_accuracy(predict_labels(w, x_val), y_val)
            if val_acc>best_acc:
                best_acc=val_acc
                best_w=copy.deepcopy(w)
                loss_best_w=copy.deepcopy(loss)
            acc_log.append(val_acc)
            print("iter:{},val_acc:{},loss:{}".format(n_iter,val_acc,loss))
    except Exception as e :
        print(e)
    return best_w,  loss_best_w ,best_acc,acc_log

