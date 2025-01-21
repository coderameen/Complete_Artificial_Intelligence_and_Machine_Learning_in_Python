#ReLU Activation Function(Rectified Linear Unit) it is an non linear activation function the ouput ranges same and given input only for +ve value input, but it give 0 for -ve values input

#FORMULA: y = f(x) = max(0,x)

def ReLU_function(x):
    return max(0,x)
print(ReLU_function(3),ReLU_function(-3),ReLU_function(-44),ReLU_function(5))#3 0 0 5