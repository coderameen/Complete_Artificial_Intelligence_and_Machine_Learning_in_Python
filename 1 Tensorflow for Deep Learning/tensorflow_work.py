#Tensorflow is just like a array/vector/matrix of n-dimension.
#Tensorflow has keras library which used for deep learning modeling
#'pip install tensorflow==2.0.0'  or  'pip install tensorflow==2.12.0'
#'pip install matplotlib'
import tensorflow as tf
print(tf.__version__)#2.12.0


##To create constant
a = tf.constant("Hello")
b = tf.constant("Prajwal")
print(type(a))#<class 'tensorflow.python.framework.ops.EagerTensor'>
print(type(b))#<class 'tensorflow.python.framework.ops.EagerTensor'>
print(a)#tf.Tensor(b'Hello', shape=(), dtype=string)
print(b)#tf.Tensor(b'Prajwal', shape=(), dtype=string)
tf.print(a)#Hello
tf.print(b)#Prajwal
tf.print(a+b)

n1 = tf.constant(5)
tf.print(n1)

#(3,3) matrix of 5
mat = tf.fill((3,3),5)
tf.print(mat)

#(3,3) matrix of 0
zero = tf.zeros((3,3))
tf.print(zero)

#(3,3) matrix of 1
ones = tf.ones((3,3))
tf.print(ones)

#2D array in tensorflow
a = tf.constant([[2,1],[1,3]])
tf.print(a)
'''
[[2 1]
 [1 3]]
'''
#shape of above
tf.print(a.get_shape())#TensorShape([2, 2])

#(2,1)=> 2 row and 1 col
b = tf.constant([[9],[2]])
tf.print(b)
tf.print(b.get_shape())#TensorShape([2, 1])


#Tensorflow variables
#String
string = tf.Variable("My name is Prajwal G", tf.string)
tf.print(string)#My name is Prajwal G

#floating
number = tf.Variable(3.14,tf.float64)
tf.print(number)#3.14

#int
my_int = tf.Variable(9,tf.int16)
tf.print(my_int)#9


#Uniformally generated Random Number in matrix
rand_mat = tf.random.uniform((5,5),0,10)
tf.print(rand_mat)
'''
[[8.22888279 2.86926031 1.33615851 0.161324739 5.20474339]
 [1.77384973 8.22144508 4.84499216 5.71008778 8.45162201]
 [2.96432734 3.16493034 6.74575806 0.383806229 0.0943958759]
 [1.7705965 4.85479355 5.86454 9.83755589 2.44160652]
 [8.35204601 4.78887939 8.42240906 5.33123732 9.07043362]]

'''
var = tf.Variable(rand_mat, tf.int16)
tf.print(var)