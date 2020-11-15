import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv("assignment6_data.csv" , comment='#')
X=np.array(df.iloc[:,0]);X=X.reshape(-1, 1)
y=np.array(df.iloc[:,1]);y=y.reshape(-1, 1)

dummy_X = np.array([-1, 0, 1])
dummy_X = dummy_X.reshape(-1,1)
dummy_Y = np.array([0, 1, 0])
dummy_Y = dummy_Y.reshape(-1,1)

#Part (i)
#Part A

X_range = np.linspace(-3,3, num=100)
X_range = X_range.reshape(-1,1)

distance_parameters = [0, 1, 5, 10, 25]
for d_param in distance_parameters:

    def gaussian_kernel(distances):
        weights = np.exp(-d_param*(distances**2))
        return weights/np.sum(weights)

    dummy_model = KNeighborsClassifier(n_neighbors=3,weights=gaussian_kernel).fit(dummy_X, dummy_Y.ravel())
    ypred = dummy_model.predict(X_range)

    plt.figure()
    plt.title("γ={} - predictions".format(d_param))
    plt.xlabel("input x"); plt.ylabel("predicted output y")
    plt.scatter(dummy_X, dummy_Y, color='red', marker='+')
    plt.plot(X_range, ypred, color='green', )
    plt.show()

#Part C

#test γ values 0-25 (as above) with range of C values .1-1000
#also report the dual_coef_parameters from the KernalRidge models

C_values = [0.1, 1, 10, 100, 1000]
distance_parameters = [1, 5, 10, 25] # wont accept a value of 0

for d_param in distance_parameters:
    for C in C_values:  
        new_model = SVC(C=C, kernel='rbf', gamma=d_param).fit(dummy_X, dummy_Y.ravel())
        ypred_new = new_model.predict(X_range)
        plt.figure()
        plt.plot(X_range, ypred_new, color='green')
        plt.xlabel("input x"); plt.ylabel("predicted output y")
        plt.scatter(dummy_X, dummy_Y, color='red', marker='+')
        plt.title("γ={} C={} - predictions".format(d_param, C))

        print("γ=", d_param, "C=", C, new_model.dual_coef_)
    plt.show()
