import numpy as np

x = [2,4,6,8]
y = [81,93,91,97]

mx = np.mean(x)
my = np.mean(y)

denominator  = np.sum([(mx - i)**2 for i in x])
numerator = np.sum([(i-mx) * (j - my) for i,j in zip(x,y)])

slope = numerator / denominator
intercept = my - (slope * mx)

print ("\n slope : {} and intercept {}".format(slope, intercept))

