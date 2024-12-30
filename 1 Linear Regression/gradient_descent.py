#wkt, Formula for Linear line y = mx + b
# m => coeffient and b => intercept

#To get BEST FIT LINE: we need to update m and b so that the cost function should be reduced in order to achieve the global minima(nothing but BEST FIT)



import numpy as np

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    learning_rate = 0.01
    n = len(x)
    iterations = 1000
    for i in range(iterations):
        #y^ = mx + b
        y_predicted = m_curr * x + b_curr
        #m derivation 
        md = -(2/n)*sum(x*(y-y_predicted))
        #b derivation
        bd = -(2/n) * sum(y-y_predicted)
        #updating m (coefficient)
        m_curr = m_curr - learning_rate * md
        #updating b (intercept)
        b_curr = b_curr - learning_rate * bd
        
        #cost funtion
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        print("m={},b={},cost={},iteration={}".format(m_curr,b_curr,cost,i))
        


x  = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)