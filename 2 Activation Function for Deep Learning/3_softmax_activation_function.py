#Softmax Activation Function: It is an Non Linear AF, which is used to generate output in range of 0 to 1
#it can used for Multi class classification
import numpy as np
def softmax_function(x):
    a = np.exp(x)
    res = a/a.sum()
    return res

print(softmax_function([3,-3,0,6]))#[4.73086069e-02 1.17266312e-04 2.35535684e-03 9.50218770e-01]
print(4.73086069e-02+1.17266312e-04+2.35535684e-03+9.50218770e-01)#1.000000000052

#Softmax action function can be used for multi class classifiction
#example remote pation, handdigit recogniton, fanshion_nmist