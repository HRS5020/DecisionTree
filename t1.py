import pandas as pd
import math

data = pd.read_csv("car.csv", header=None)
c_name =[]
for i in data:
    c_name.append("att" + str(i))

data.columns = c_name

print(data["att5"].unique())

d1 = data[data["att5"] == 'high'].copy()
print(d1.columns)