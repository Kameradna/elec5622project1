import numpy as np
import os
import argparse
from csv import reader
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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
    tuple_len = raw_data.shape()[1]-1
    for row in raw_data:
        data_x.append([j for j in row[0:tuple_len]])
        data_y.append(int(row[tuple_len]))
    return data_x, data_y


def main(args):
    if args.dummy:
        x_train, y_train, x_test, y_test = make_dummy_data(args)
    else:
        data_x, data_y = preprocess("dprk.csv")
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=args.test_size, random_state=None)
    
    print("Fitting SVM model...")
    fitted_model = make_pipeline(StandardScaler(),PCA(n_components=args.pca_components),svm.SVC(C=args.C,kernel=args.kernel,degree=args.degree,gamma=args.gamma,verbose=args.verbose)).fit(x_train,y_train)
    train_score = fitted_model.score(x_train, y_train)
    test_score = fitted_model.score(x_test, y_test)
    print(f'The training accuracy of the trained SVM is {100*train_score:.2f}%')
    print(f'The testing accuracy of the trained SVM is {100*test_score:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_components", type=int, default=None, help="PCA components to reduce towards")

    parser.add_argument("--C", type=float, default=1.0, help="SVM parameter C")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel")
    parser.add_argument("--degree", type=int, default=3, help="SVM degree")
    parser.add_argument("--gamma", default="scale", help="SVM gamma, takes str or float")
    parser.add_argument("--verbose",default=False,action="store_true", help="SVM verbose output?")

    parser.add_argument("--dummy",default=False,action="store_true", help="use dummy data?")
    parser.add_argument("--dummy_num", type=int, default=100, help="amount of dummy data to produce")
    parser.add_argument("--dummy_dim", type=int, default=100, help="dimension of dummy data to produce")
    args = parser.parse_args()
    main(args)
    

