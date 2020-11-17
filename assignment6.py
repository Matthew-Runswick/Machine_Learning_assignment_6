import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import statistics

df = pd.read_csv("assignment6_data.csv" , comment='#')
X=np.array(df.iloc[:,0]);X=X.reshape(-1, 1)
y=np.array(df.iloc[:,1]);y=y.reshape(-1, 1)

dummy_X = [[-1], [0], [1]]
dummy_Y = [0, 1, 0]

#Part (i)
#Part A

X_range = np.linspace(-3,3, num=100)
X_range = X_range.reshape(-1,1)

# distance_parameters = [0, 1, 5, 10, 25]
# for d_param in distance_parameters:

#     def gaussian_kernel(distances):
#         weights = np.exp(-d_param*(distances**2))
#         return weights/np.sum(weights)

#     dummy_model = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel).fit(dummy_X, dummy_Y)
#     ypred = dummy_model.predict(X_range)

#     plt.figure()
#     plt.title("γ={} - predictions".format(d_param))
#     plt.xlabel("input x"); plt.ylabel("predicted output y")
#     plt.scatter(dummy_X, dummy_Y, color='red', marker='+')
#     plt.plot(X_range, ypred, color='green', )
#     plt.show()

# #Part C
# C_values = [0.1, 1, 10, 100, 1000]
# distance_parameters = [0, 1, 5, 10, 25] # wont accept a value of 0

# for d_param in distance_parameters:
#     for C in C_values:
#         new_model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=d_param).fit(dummy_X, dummy_Y)
#         ypred_new = new_model.predict(X_range)
#         plt.figure()
#         plt.plot(X_range, ypred_new, color='green')
#         plt.xlabel("input x"); plt.ylabel("predicted output y")
#         plt.scatter(dummy_X, dummy_Y, color='red', marker='+')
#         plt.title("γ={} C={} - predictions".format(d_param, C))

#         print("γ=", d_param, "C=", C, new_model.dual_coef_)
#     plt.show()

# #part (ii)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) *************(potentially need to use, talk to others)*****************

# #Part A
# distance_parameters = [0, 1, 5, 10, 25, 50]
# kf = KFold(n_splits = 5)
# mean_values_KNeighbors = []
# standard_deviation_values_KNeighbors =[]
# predictions_KNeighbors = []

# for d_param in distance_parameters:
#     new_estimates = []
#     for train, test, in kf.split(X):
#         def gaussian_kernel(distances):
#             weights = np.exp(-d_param*(distances**2))
#             return weights/np.sum(weights)

#         new_model = KNeighborsRegressor(n_neighbors=len(train),weights=gaussian_kernel).fit(X[train], y[train])
#         test_pred = new_model.predict(X[test])
#         new_estimates.append(mean_squared_error(test_pred, y[test]))

#     predictions_KNeighbors.append(new_model.predict(X_range))

#     mean = sum(new_estimates)/6
#     standard_deviation = statistics.stdev(new_estimates)
#     mean_values_KNeighbors.append(mean)
#     standard_deviation_values_KNeighbors.append(standard_deviation)

# plt.figure()
# plt.title("predictions over real data for various γ values")
# plt.xlabel("input x"); plt.ylabel("predicted output y")
# plt.scatter(X, y, color='red', marker='+')
# col_and_label = [["red", "γ=0"], ["green", "γ=1"], ["black", "γ=5"], ["blue", "γ=10"], ["yellow", "γ=25"], ["cyan", "γ=50"]]
# i = 0
# for ypred in predictions_KNeighbors:
#     plt.plot(X_range, ypred, color=col_and_label[i][0], label=col_and_label[i][1] )
#     i = i + 1
# plt.legend()
# plt.show()

# print("mean values", mean_values_KNeighbors)
# print("standard deviation values", standard_deviation_values_KNeighbors)
# fig = plt.figure()
# plt.title("γ Value vs Average Mean Squared Errors With Standard Deviation")
# plt.errorbar(distance_parameters, mean_values_KNeighbors, yerr=standard_deviation_values_KNeighbors)
# plt.xlabel("gamma Values")
# plt.ylabel("Average of Mean Squared Errors")
# plt.show()

# part B
distance_parameters = [0, 1, 5, 10, 25] # wont accept a value of 0
C_values = [0.1, 1, 10, 100, 1000]

for d_param in distance_parameters:
    kf = KFold(n_splits = 5)
    mean_values_KernelRidge = []
    standard_deviation_values_KernelRidge =[]
    predictions_KernelRidge = []
    for C in C_values:
        new_estimates = []
        for train, test, in kf.split(X):
            new_model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=d_param).fit(X[train], y[train])
            ypred_new = new_model.predict(X_range)
            test_pred = new_model.predict(X[test])
            new_estimates.append(mean_squared_error(test_pred, y[test]))

        predictions_KernelRidge.append(new_model.predict(X_range))

        mean = sum(new_estimates)/5
        standard_deviation = statistics.stdev(new_estimates)
        mean_values_KernelRidge.append(mean)
        standard_deviation_values_KernelRidge.append(standard_deviation)

    plt.figure()
    plt.title("predictions over real data for various C values for γ={}".format(d_param))
    plt.xlabel("input x"); plt.ylabel("predicted output y")
    plt.scatter(X, y, color='red', marker='+')
    col_and_label = [["red", "C=0.1"], ["green", "C=1"], ["black", "C=10"], ["blue", "C=100"], ["yellow", "C=1000"]]
    i = 0
    for ypred in predictions_KernelRidge:
        plt.plot(X_range, ypred, color=col_and_label[i][0], label=col_and_label[i][1] )
        i = i + 1
    plt.legend()
    plt.show()

    # print("mean values", mean_values_KernelRidge)
    # print("standard deviation values", standard_deviation_values_KernelRidge)
    fig = plt.figure()
    plt.title("C Value vs Average Mean Squared Errors With Standard Deviation")
    plt.errorbar(C_values, mean_values_KernelRidge, yerr=standard_deviation_values_KernelRidge)
    plt.xscale("log")
    plt.xlabel("gamma Values");plt.ylabel("Average of Mean Squared Errors")
    plt.show()



#questions for dave
#do u split data 20/80 test/training
#calculating error over real values not over full X_range (-3, 3)
#do u check every gamma value for every C value 5 times (a lot of graphs)
#do u do kfold for both the C values and the gamma values?