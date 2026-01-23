import numpy as np
def load_data(filename):
    data = np.loadtxt(filename,delimiter=",")
    X = data[:,:2]
    y = data[:,2]
    return X,y

def sigmoid(z):
    return 1/(1 + np.exp(-z))



def map_feature(x1, x2, degree = 6):
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)


    out = []

    for i in range(1, degree + 1):
        for j in range(i+1):
            out.append((x1**(i-j) * (x2**j)))
    return np.stack(out, axis = 1)



        