# # lecture 12 excercise 1
import numpy as np

# v = np.array([2., 2., 4.])

# e0 = np.array([1.,0.,0.])
# e1 = np.array([0.,1.,0.])
# e2 = np.array([0.,0.,1.])

# projection_e0 = np.dot(v,e0)
# projection_e1 = np.dot(v,e1)
# projection_e2 = np.dot(v,e2)

# print(projection_e0)
# print(projection_e1)
# print(projection_e2)


# # lecture 12 excercise 2
# matrix1 = np.array([[6,-9,1],[4,24,8]])
# print(2*matrix1)
# matrix2 = np.array([[1,0],[0,1]]) # np.identity(2)
# matrix3 = np.array([[6,-9,1],[4,24,8]])
# print(np.dot(matrix2,matrix3))
# matrix4 = np.array([[4,3],[3,2]])
# matrix5 = np.array([[-2,3],[3,-4]])
# print(np.dot(matrix4,matrix5)) #identity matrix -> 4 is 5 inverse and 5 is 4 inverse


# # lecture 12 excercise 3
# def swapRows(M,a,b):
#     M1 = np.array(M[:,a])
#     M[:,a] = M[:,b]
#     M[:,b] = M1
#     return M

# def swapCols(M,a,b):
#     M1 = np.array(M[a,:])
#     M[a,:] = M[b,:]
#     M[b,:] = M1
#     return M

# M = np.array([[93,  95],
#               [84, 100],
#               [99,  87]])

# print(swapRows(M,0,1))
# print(swapCols(M,1,2))


# #lecture 12 excercise 4
# def set_array(L,rows,cols):
#     L = np.array(L)
#     L = L.reshape(rows,cols)
#     return L

# list = [1,2,3,4,5,7]
# print(set_array(list,2,3))

#lecture 12 excercise 5
array = np.array[[0, 1, 2, 3, 4, 5],
                 [10, 11, 12, 13, 14, 15],
                 [20, 21, 22, 23, 24, 25],
                 [30, 31, 32, 33, 34, 35],
                 [40, 41, 42, 43, 44, 45],
                 [50, 51, 52, 53, 54, 55]]

