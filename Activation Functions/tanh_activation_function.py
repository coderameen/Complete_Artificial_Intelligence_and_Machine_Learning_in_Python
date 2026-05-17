#Tanh Activation Function: it is Non Linear AF, used to generate ouput within the range of -1 to 1
#FORMULA = 2Sigmoid(2x)-1

#sigmoid = 1/(1+e^-x)

import numpy as np

def tanh_function(x):
    return 2 * (1/(1+np.exp(-2*x))) -1

print(tanh_function(3),tanh_function(-3),tanh_function(0),tanh_function(6))#0.9950547536867307 -0.9950547536867305 0.0 0.9999877116507956

#NOTE: The output ranges within from -1 to 1