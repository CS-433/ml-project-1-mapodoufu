""" Module containing all implementations of ML techniques required for the project """

import numpy as np
import csv
import copy


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


def compute_loss(y, x, w, mae=False):

    e = y - x @ w
    if mae:
        loss = np.mean(np.abs(e))
    else:
        loss = np.mean(e ** 2) / 2
    return loss


def compute_gradient_mse(y, x, w):

    n = x.shape[0]
    e = y - x @ w
    grd = -(x.T @ e) / n
    return grd


def compute_subgradient_mae(y, x, w):
    n = x.shape[0]
    e = y - x @ w
    grd = -(x.T @ np.sign(e)) / n
    return grd


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


def compute_gradient_hinge(y, x, w, lambda_=0):

    mask = (y * (x @ w)) < 1
    grad = np.zeros_like(w)
    grad -= x.T @ (mask * y)
    grad += lambda_ * w
    return grad

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
        print(x[:, std_x == 0]) # std = 0, data are all identical and equal to the mean. Unlikely!
    x_norm = (x - mean_x) / std_x
    return x_norm, mean_x, std_x



def split_data_by_categories( x,y, ids, PRI_JET_NUM_index):


    ## divide the data into 3 subsets
    category_list =[]
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


def build_poly_and_cross(x, degree=2, cross_term=True):

    n, d = x.shape
    powers = [x ** deg for deg in range(1, degree + 1)]
    phi = np.concatenate((np.ones((n, 1)), *powers), axis=1)
    if cross_term:
        new_feat = np.array([x[:, i] * x[:, j] for i in range(d) for j in range(i + 1, d)]).T
        phi = np.append(phi, new_feat, axis=1)
    # Brainstorm: you can even append more features, such as sqrt(x_i), log(x_i), just do the same as above!
    return phi


def preprocess_data(data, nan_value=-999., low_var_threshold=0.1, corr_threshold=0.9,
                           degree=2, cross_term=True, columns_to_remove=None, norm_first=True, mean=None, std=None):

    data = data.copy()
    data[data == nan_value] = np.nan # replace -999. to np.nan
    data = remove_nan_columns(data) # Remove coulmns that are filled with NA or 0 or (NA and 0), still possible we have columns with some NA and 0
    
    """"""
    if columns_to_remove is not None: ## for test dataset
        data = np.delete(data, columns_to_remove, axis=1)
    else:   ## for train dataset
        data, columns_to_remove = remove_correlated_features(data, corr_threshold) # Remove correlating columns

    data = build_poly_and_cross(data, degree, cross_term) # cross_term: if we want terms not only x_i^(pow), but also x_i * x_j
    
    ## normalize
    data[:, 1:], mean, std = z_normalize_data(data[:, 1:], mean, std)
    data[np.isnan(data)] = 0. # For NA values, simply replace them as 0
    return data, columns_to_remove, mean, std



def reg_logistic_regression_with_val(y_tr, x_tr,y_val, x_val, lambda_, initial_w, max_iters, gamma, threshold=1e-2):

    # Map classes from {-1, 1} to {0, 1}
    y_tr = map_target_classes_to_bool(y_tr)

    w = initial_w

    best_acc=-1
    best_w=None
    try :   
        for n_iter in range(max_iters):
            # Compute the gradient and Hessian of the loss function
            grd = compute_gradient_reg_regress(y_tr, x_tr, w, lambda_)
            hess = compute_hessian_reg_regress(y_tr, x_tr, w, lambda_)

            # Update the weights
            w -= gamma / np.sqrt(n_iter+1) * np.linalg.solve(hess, grd)

            # Compute the current loss and test convergence
            loss = compute_loss_reg_regress(y_tr, x_tr, w, lambda_)

            val_acc=compute_accuracy(predict_labels(w, x_val), y_val)
            if val_acc>best_acc: # Minimize the test error while training
                best_acc=val_acc
                best_w=copy.deepcopy(w)
                loss_best_w=copy.deepcopy(loss)
            print("iter:{},val_acc:{},loss:{}".format(n_iter,val_acc,loss))
    except Exception as e :
        print(e)
    return best_w,  loss_best_w ,best_acc

