import numpy as np
import json

def parse_array2(d):
    (rows, cols) = d['dim']
    data = d['data']

    a = np.zeros((rows, cols))

    for i in range(0, rows):
        k = i * cols

        for j in range(0, cols):
            a[i][j] = data[k + j]

        #a[i][0] = data[k]
        #a[i][1] = data[k+1]
        #a[i][2] = data[k+2]

    return a

def parse_array1(d):
    dim = d['dim']
    data = d['data']

    a = np.zeros(len(data))

    for i in range(len(a)):
        a[i] = data[i]

    return a
