import numpy as np

slope = 3
intercept = 76

data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    val = []
    for i in x:
        val.append(slope * i + intercept)
    return val

def rmse(p,a):
    return np.sqrt(((p - a) ** 2).mean())

predict_result = predict(x)

print("rmse : {}".format(rmse(np.array(predict_result), np.array(y))))