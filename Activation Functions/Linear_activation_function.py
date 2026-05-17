#Activation Function: it is an function used to generate output via Formula
'''
Type of Activation Function
1.Linear Activation Function
2.Non Linear Activation Function(EX: Sigmoid AF,Softmax AF,Tanh AF,ReLU AF)
'''

#1.Linear Activation Function: It is a fuction where generated output is same as give input
#The graph is Stright line
#Formula: y = f(x) = x
#f(4) = 4, f(-3) = -3, f(0) = 0
def linear_function(x):
    return x
print(linear_function(3), linear_function(-3),linear_function(6))

#NOTE: the ouput is same as given input