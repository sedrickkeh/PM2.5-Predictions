import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from baseline import nn_model
from wavelet import wavelet_model
from cluster import cluster_model


def get_train_test(args):
    df = pd.read_csv("features_final.csv")

    y = df["ug/m3"]
    y = np.array(y)

    if (args.hybrid): X = np.array([df["%"], df["Degree Celsius_x"], df["Degree Celsius_y"], df["m/s"], df["Day of Week"], df["Normalized Day of Year"], df["Previous Concentration"], df["Prediction Ratio"]])
    else: X = np.array([df["%"], df["Degree Celsius_x"], df["Degree Celsius_y"], df["m/s"], df["Day of Week"], df["Normalized Day of Year"], df["Previous Concentration"]])
    X = np.transpose(X)

    a = df["Degree Celsius_y"]

    if (args.normalize):
        y = y / y.max(axis=0)
        X = X / X.max(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=52)
    return X_train, X_test, y_train, y_test


def plot_points(actual, pred):
    plt.scatter(actual, pred)
    plt.title('Actual vs Predicted PM2.5 Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_iter", type=int, default=800)
    parser.add_argument("--layers", type=tuple, default=(64, 128, 100))
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("-hybrid", "--hybrid", action='store_true')
    parser.add_argument("-normalize", "--normalize", action='store_true')
    args = parser.parse_args()

    cntr = 0
    for i in range(args.num_test):
        X_train, X_test, y_train, y_test = get_train_test(args)

        # Mode 0 = baseline model
        # Mode 1 = wavelet model
        # Mode 2 = clustering model
        if (args.mode == 0):
            y_pred = nn_model(X_train, X_test, y_train, args.lr, args.max_iter, args.layers)
        elif (args.mode == 1):
            y_pred = wavelet_model(X_train, X_test, y_train, args.lr, args.max_iter, args.layers)
        elif (args.mode == 2):
            if (args.clusters <= 0):
                print("Please enter a valid number of clusters.")
                return
            y_pred = cluster_model(X_train, X_test, y_train, args.lr, args.max_iter, args.layers, args.clusters)
        else:
            print("Please enter valid mode (0) - baseline, (1) - wavelet, (2) - clustering")
            return

        acc = r2_score(y_test, y_pred)
        print("Test number {}, Test accuracy: {}".format(i, acc))
        cntr += acc

        if (i == args.num_test-1):
            plot_points(y_test, y_pred)

    print("Final accuracy: {}".format(cntr / args.num_test))

if __name__=="__main__":
    main()