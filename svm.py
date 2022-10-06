import numpy as np
import os
import argparse
from csv import reader
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate


def make_dummy_data(args):
    num = args.dummy_num
    dim = args.dummy_dim
    raw_data = np.random.rand(num,dim)
    outcomes = np.round(np.random.rand(num,1))
    raw_data = np.concatenate((raw_data,outcomes),axis=1)

    data_x = []
    data_y = []
    tuple_len = raw_data.shape[1]-1
    for row in raw_data:
        data_x.append([j for j in row[0:tuple_len]])
        data_y.append(int(row[tuple_len]))

    split = int(np.floor(num*9/10))
    x_train, y_train = data_x[:split], data_y[:split]
    print(f"We have {len(x_train)} training vectors with dim {len(x_train[0])}")
    if len(x_train) < len(x_train[0]):
        print("We may be experiencing the small sample size issue")
    x_test, y_test = data_x[split:], data_y[split:]
    return x_train, y_train, x_test, y_test


def preprocess(path="dprk.csv"):
    with open(path) as f:
        csv_data = reader(f, delimiter=",")
        raw_data = np.array(list(csv_data))
    data_x = []
    data_y = []
    names = []
    tuple_len = raw_data.shape[1]
    for row in raw_data:
        data_x.append([j for j in row[2:tuple_len:2]])#grabbing every second column so we capture only the mm cubed readings
        data_y.append(1 if row[0][0:2] == "AD" else -1)
        names.append(row[0])
    # print(data_x, data_y)
    # print(len(data_x),len(data_x[0]))
    # print(len(data_y))
    return data_x, data_y, names


def main(args):
    if args.dummy:
        x_train, y_train, x_test, y_test = make_dummy_data(args)
    else:
        data_x, data_y, _ = preprocess(args.path)
        if args.val_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=args.val_size, random_state=None,stratify=data_y)
        else:
            x_train, y_train = data_x, data_y
    
    print("Fitting SVM model...")
    print(f"Using {len(x_val)} validation items out of {len(data_x)}")
    model = make_pipeline(StandardScaler(),PCA(n_components=args.pca_components),svm.SVC(C=args.C,kernel=args.kernel,degree=args.degree,gamma=args.gamma,verbose=args.verbose))

    if args.k_fold is not None:#not working
        model = svm.SVC(C=args.C,kernel=args.kernel,degree=args.degree,gamma=args.gamma,verbose=args.verbose)
        scores = cross_validate(model, data_x, y=data_y, scoring=model.score(data_x,data_y), cv=None, n_jobs=-1, verbose=1, fit_params={_:data_x,_:data_y}, pre_dispatch='2*n_jobs', return_train_score=True, return_estimator=True, error_score='raise')
        print(scores)
        exit(1)
    
    fitted_model = model.fit(x_train,y_train)
    train_score = fitted_model.score(x_train, y_train)
    print(f'The training accuracy of the trained SVM is {100*train_score:.2f}%')
    if args.val_size > 0:
        validation_score = fitted_model.score(x_val, y_val)
        print(f'The validation accuracy of the trained SVM is {100*validation_score:.2f}%')

    print("# Testing phase #")
    if args.test_path is not None:
        test_x, _, names = preprocess(args.test_path)
        y_pred = fitted_model.predict(test_x)
        predictions = {}
        for idx, name in enumerate(names):
            predictions[name] = y_pred[idx]
        print(predictions)

    if args.multi_run > 0:
        aggregate_predictions = {}
        for idx, name in enumerate(names):
            aggregate_predictions[name] = []

        for i in range(args.multi_run):#rep this args.multi_run times
            for args.val_size in range(len(data_x)//2):
                for args.pca_components in range(1, len(data_x)//2):
                    '''
                    Resplit the data each run, then fit and test the data, recording results.
                    '''
                    if args.val_size > 1:
                        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=args.val_size, random_state=None,stratify=data_y)
                    else:
                        x_train, y_train = data_x, data_y
                    model = make_pipeline(StandardScaler(),PCA(n_components=args.pca_components),svm.SVC(C=args.C,kernel=args.kernel,degree=args.degree,gamma=args.gamma,verbose=args.verbose))
                    fitted_model = model.fit(x_train,y_train)
                    y_pred = fitted_model.predict(test_x)
                    for idx, name in enumerate(names):
                        aggregate_predictions[name].append(y_pred[idx])
        
        aggregate_predictions = {name: np.mean(aggregate_predictions[name]) for name in aggregate_predictions}
        print(aggregate_predictions)

        for name in aggregate_predictions:
            print(f"{name}, {aggregate_predictions[name]}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_components", type=int, default=None, help="PCA components to reduce towards")

    parser.add_argument("--C", type=float, default=1.0, help="SVM parameter C")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel")
    parser.add_argument("--degree", type=int, default=3, help="SVM degree")
    parser.add_argument("--gamma", default="scale", help="SVM gamma, takes str or float")
    parser.add_argument("--verbose",default=False,action="store_true", help="SVM verbose output?")

    parser.add_argument("--val_size", type=float, default=2, help="validation set size")

    parser.add_argument("--path", type=str, default="dprk.csv", help="relative path to csv")
    parser.add_argument("--test_path", type=str, default=None, help="relative path to csv")
    parser.add_argument("--multi_run", type=int, default=0, help="multi runs and report the most popular")
    parser.add_argument("--k_fold", type=int, default=None, help="k-fold cross validation?")

    parser.add_argument("--dummy",default=False,action="store_true", help="use dummy data?")
    parser.add_argument("--dummy_num", type=int, default=100, help="amount of dummy data to produce")
    parser.add_argument("--dummy_dim", type=int, default=100, help="dimension of dummy data to produce")
    args = parser.parse_args()
    main(args)