#Sigmoid Activation Function: it is an Non Linear Activation, where the ouput of the function ranges from 0 to 1
#Graph line is S-shaped line
#whatever the input is, the output ranges from 0 to 1
#Sigmoid is only used for 2 class classification
#FORMULA: 1/(1+e^-x)

import numpy as np
def sigmoid_function(x):
    return 1/(1+np.exp(-x))

print(sigmoid_function(3),sigmoid_function(-3))#0.9525741268224334 0.04742587317756678
print(0.9525741268224334 + 0.04742587317756678)#1.0000000000000002

# print(sigmoid_function(3),sigmoid_function(-3), sigmoid_function(6))#0.9525741268224334 0.04742587317756678 0.9975273768433653

#NOTE:Sigmoid AF is only used to make 2 class classification
#NOTE: Sigmoid AP can't be used to make multi class classsificaiton
#NOTE: With the help of Softmax AF we can achive for multi class classfication