import os
import csv 
import numpy as np
import math

cwdir = os.getcwd()
addon = "Documents\\magnet\\magnet"
basepath = os.path.join(cwdir,addon)
scalar = 0.2
MagnetData = os.path.join(cwdir, "Documents\MagnetData.csv")
Switch = False
if Switch:
    while scalar <= 2.0:
        magpath = f"{scalar:.1f}\{scalar:.1f}test.csv"
        datapath = os.path.join(basepath, magpath)
        file = open(MagnetData, 'a+')
        h0data = np.genfromtxt(datapath, delimiter=',', skip_header=(1), usecols=(3))
        h1data = np.genfromtxt(datapath, delimiter=',', skip_header=(1), usecols=(4))
        h2data = np.genfromtxt(datapath, delimiter=',', skip_header=(1), usecols=(5))
        hmag = np.sqrt((h0data*h0data)+(h1data*h1data)+(h2data*h2data))
        hmag = hmag.tolist()
        hmagpolished = [scalar] + hmag
        print(hmagpolished)
        with file: 
            write = csv.writer(file)
            write.writerow(hmagpolished)
        scalar = scalar + 0.2

# Getting x-cords of the points along with the actual sim data for 0.4 and 1.6
xpath = os.path.join(basepath, "1.0\\1.0test.csv")
xposition = np.genfromtxt(xpath, delimiter=',',skip_header=(1), usecols=(10))

################# Regression Timers and Accuracy ###################### 
from lazypredict.Supervised import LazyRegressor
import seaborn as sns; sns.set()
from sklearn.utils import shuffle

# This function will take in data and run all kinds of ML techniques on them and output the time that it took to run and a basic evaluation metric
def MLtimers(X, y):
    X, y = shuffle(X, y, random_state=13)
    X = X.astype(np.float32)

    offset = int(X.shape[0] * 0.9)

    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    print(models)