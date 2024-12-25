#Numpy is an numerical python used for data analytics and scientific computing in python"
#NOTE: list is slow then numpy
#pip install numpy

import numpy as np

#one dimensional array
arr = np.array([10,20,30,40])
print(arr)
#To define datatype (Note default int is the datatype= int8 or int16 or int32 or int64)
arr = np.array([2,3,6,7],dtype='float')
print(arr)
print(arr.dtype)
print(arr.shape)

#2D array#matrix
arr = np.array([[5,6,7],[8,9,2]])
print(arr)
print(arr.dtype)
print(type(arr))#<class 'numpy.ndarray'>
print(arr.size)#take count of items in array


#accessing array items
#1d array
arr1 = np.array([10,20,33,44])
print(arr1[2])#33

#2d array
arr2 = np.array([[5,6,7],[8,9,2]])
print(arr2[1][0])#8
#can I apply slicing arr[row,col]
print(arr2[:,0])#[5 8]

#3d array
a = np.array([[[5,7,8],[6,8,9]],[[6,8,9],[1,2,3]]])
print(a)

#Type of matrix\(numpy)
#1.zero matrix
z = np.zeros((4,3))
print(z)
print(z.dtype)# float64
z = np.zeros((4,3),dtype='int32')
print(z)
print(z.dtype)#int32
'''
[[0 0 0]
 [0 0 0]
 [0 0 0]
 [0 0 0]]
'''



#2.ones matrix
o = np.ones((4,4),dtype='int16')
print(o)
print(o.dtype)#int16
'''
[[1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]]

'''

#3.identity matrix
arr = np.identity(4, dtype='int32')#4=> (4,4)
print(arr)
#diagnol elements are 1
'''
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 0 1]]
'''

#4.full and full_like
arr = np.full((4,4),6)
print(arr)
'''
[[6 6 6 6]
 [6 6 6 6]
 [6 6 6 6]
 [6 6 6 6]]
'''


arr = np.full_like(arr,2)
print(arr)
'''

[[2 2 2 2]
 [2 2 2 2]
 [2 2 2 2]
 [2 2 2 2]]
'''

#5.random matrxi
mat = np.random.rand(3,2)
print(mat)
'''
[[0.25609784 0.46246604]
 [0.89711978 0.86875038]
 [0.90068322 0.90000017]]
'''

#6.random item in range
arr = np.random.randint(1,10,(4,4))
print(arr)
'''
[[3 4 6 1]
 [7 3 3 7]
 [4 7 4 4]
 [8 1 9 8]]
'''


arrp = np.ones((4,4))
print(arrp)
# arrs = arrp#Note not to use
arrs = arrp.copy()
print(arrs)


#Arithemetic operation with numpy

arr = np.array([2,1,4,5])
print(arr)#[2 1 4 5]

print(arr + 2)#[4 3 6 7]
print(arr*2)#[ 4  2  8 10]
print(arr-2)
print(arr//2)

arr = np.sin(arr)
print(arr)#[ 0.90929743  0.84147098 -0.7568025  -0.95892427]

arr = np.cos(arr)
print(arr)#[0.61430028 0.66636675 0.72703513 0.57440088]