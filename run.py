import numpy as np

from implementations import load_csv_data, preprocess_data, split_data_by_categories,predict_labels, create_csv_submission, compute_accuracy,reg_logistic_regression_with_val

PRI_JET_NUM_INDEX = 22
SEED = 114514
import random
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(SEED)
random.seed(SEED)
## funciton definition
def split_data(x, y, ratio):
    """split the dataset based on the split ratio."""
    # set seed
    #
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, y_tr, x_te, y_te

if __name__ == "__main__":

    ############################
    #
    # LOAD .csv file
    #
    ############################
    x_train,y_train,ids_train = load_csv_data("data/train.csv")
    x_test,y_test,ids_test = load_csv_data("data/test.csv")

    ############################
    #
    # split data by categories
    #
    ############################
    x_train_split, y_train_split, ids_train_split = \
        split_data_by_categories(x_train,y_train,ids_train,PRI_JET_NUM_INDEX)
    x_test_split, y_test_split, ids_test_split = \
        split_data_by_categories(x_test,y_test, ids_test,PRI_JET_NUM_INDEX)
    
    ############################
    #
    # load params selected by grid searching
    #
    ############################
    
    
    logistic_best_params = np.array([[ 0,   11,    0.2 ],
        [ 0,   12,    0.5 ],
        [ 0,   10,    0.5 ]
        ])
    """
    import sys
    logistic_best_params = np.array([[ float(sys.argv[1]),  int(sys.argv[2]),    float(sys.argv[3]) ],
        [ float(sys.argv[4]),   int(sys.argv[5]),    float(sys.argv[6]) ],
        [ float(sys.argv[7]),   int(sys.argv[8]),    float(sys.argv[9]) ]])        
    """
    ############################
    #
    # train
    #
    ############################
    logistic_best_models = []
    cross_term=True
    acc_logs=[]
    for (lambda_, deg, gamma), train_data_split, train_classes_split  in \
       zip(logistic_best_params, x_train_split,      y_train_split):

        model_index=len(logistic_best_models)## the index of the model (total 3 models)
        print("training: model_index:",model_index,"params:",lambda_, deg, gamma)
        data_split, columns_to_remove, mean, std = preprocess_data(train_data_split, degree=np.int(deg),
                                                                          cross_term=cross_term)

        
        x_tr, y_tr, x_val, y_val=split_data(data_split, train_classes_split,0.8)
        print("shape:",x_tr.shape,y_tr.shape,x_tr.shape,y_val.shape)
        initial_w = np.zeros((data_split.shape[1],))
        print("w shape:",initial_w.shape)
        w, loss,val_acc,acc_log= reg_logistic_regression_with_val(y_tr, x_tr , y_val, x_val, lambda_, initial_w, 300, gamma, 1)
        acc_logs.append(acc_log)
        #print('Loss:{} Acc :{}'.format(loss,compute_accuracy(predict_labels(w, data_split), train_classes_split)) )
        print('Loss:{} Acc :{}'.format(loss,val_acc) )

        file_handle=open('log_res.txt',mode='a')
        file_handle.write('model_index:{},lambda_:{},deg:{},gamma:{},val_acc:{},\n'.format(model_index,lambda_, deg, gamma,val_acc))
        file_handle.close()

        logistic_best_models.append((w, loss, columns_to_remove, mean, std))
    ############################
    #
    # visualization
    #
    ############################
    plt.rc('lines', linewidth=2.5)
    fig, ax = plt.subplots()
    line_x=[np.linspace(0, len(x),len(x) ) for x in acc_logs]
    lines=[]
    line_styles=['-',':','-.']
    colors=['pink','red','green']
    #print(acc_logs)
    for i in range(len(acc_logs)):
         lines.append(ax.plot(line_x[i], acc_logs[i],color=colors[i],linestyle=line_styles[i], label='Category {}'.format(i))) 
    
    ax.legend()#'lower center'
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    plt.savefig('acc.png')
    #plt.show()

    ############################
    #
    # prediction
    #
    ############################
    results = None
    for (w, _, col_to_rm, mean, std), (_, deg, _),        test_data_split,  test_classes_split,  test_ids_split in \
     zip(logistic_best_models,    logistic_best_params,    x_test_split,     y_test_split,    ids_test_split):
        test_data_split, _, _, _ = preprocess_data(test_data_split, degree=np.int(deg),
                                                          columns_to_remove=col_to_rm,
                                                          cross_term=cross_term, norm_first=False, mean=mean, std=std)
        pred = predict_labels(w, test_data_split)
        out = np.stack((test_ids_split, pred), axis=-1)
        results = out if results is None else np.vstack((results, out))

    # Create the submission
    create_csv_submission(results[:, 0], results[:, 1], 'results/submission.csv')
