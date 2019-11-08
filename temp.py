import argparse
import numpy as np
import pandas as pd
import math

data = pd.read_csv("car.csv", header=None)

# column names added  (att0, att1, att2 ......)
c_name = []
for i in data:
    c_name.append("att" + str(i))
data.columns = c_name
del c_name[-1]

# calculate the logbase (number of unique values in target column)
b1 = list(data["att6"].value_counts())
logbase = len(b1)

# System Entropy calculation
overallentropy = 0
for i in b1:
    temp = (-1) * (i / sum(b1)) * (math.log((i / sum(b1)), logbase))
    overallentropy += temp
print(overallentropy)


# node selection for decision tree
def nodeselector(temp_data):
    b2 = list(temp_data["att6"].unique())
    column_info_gain = {}
    for column_name in c_name:
        a = list(temp_data[column_name].unique())
        columnEntropy = 0
        for i in range(len(a)):
            t1 = temp_data[temp_data[column_name] == a[i]]
            sum = 0
            for j in range(len(b2)):
                t2 = t1[t1["att6"] == b2[j]]
                if t2.shape[0] == 0:
                    e = 0
                else:
                    e = (-1) * (t2.shape[0] / t1.shape[0]) * (math.log(t2.shape[0] / t1.shape[0], logbase))
                sum = sum + e
            columnEntropy += sum * (t1.shape[0] / data.shape[0])
        print("entropy of column {0}  =  {1}".format(column_name, str(columnEntropy)))
        print("----------------------------------------")
        column_info_gain.update({column_name: overallentropy - columnEntropy})
    print(column_info_gain)
    print(max(column_info_gain))
    selected_node = max(column_info_gain)
    selected_node_unique = list(temp_data[max(column_info_gain)].unique())

    d = {}
    for i in range(len(selected_node_unique)):
        print(i)

        d1 = temp_data[temp_data[selected_node] == selected_node_unique[i]].copy()
        d["value{0}".format(i)] = d1

    print(d)






nodeselector(data)
