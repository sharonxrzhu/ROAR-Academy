# # lecture 12 excercise 1
#import numpy as np

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
#     if M.shape[0] < a or M.shape[0] < b:
#         M1 = np.array(M[:,a])
#         M[:,a] = M[:,b] #M[[a,b]] = M[[b,a]]
#         M[:,b] = M1
#         return M

# def swapCols(M,a,b):
#     M1 = np.array(M[a,:]) #M[:,[a,b]] = M[:,[b,a]]
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

# #Tianlun's solution
# def set_array(L, rows, cols, order = "row-col"):
#     try:
#         rows = int(rows)
#         cols = int(cols)
#     except:
#         print("invalid input")
#         return None
    
#     if len(L) != rows * cols:
#         print("invalid dimension")
#         return None
#     order = order.lower()
#     if order == 'row-col':
#         return L.reshape(rows,cols)
#     elif order == "col-row":
#         return L.reshape(cols, rows)
#     else:
#         print("invalid order")
#         return None


# #lecture 12 excercise 5
# arr = np.array([[0, 1, 2, 3, 4, 5],
#                  [10, 11, 12, 13, 14, 15],
#                  [20, 21, 22, 23, 24, 25],
#                  [30, 31, 32, 33, 34, 35],
#                  [40, 41, 42, 43, 44, 45],
#                  [50, 51, 52, 53, 54, 55]])
# print(arr[:,[1]])
# print(arr[1,2:4])
# print(arr[2:4,4:6])


# #lecture 11 excercise 1
# file = open("motto.txt",'w')
# file.write('Fiat Lux!')

# file = open("motto.txt",'a+')
# content = file.read()
# print(content)
# file.write("Let there be light")

# file.close()

# # lecture 11 excercise 2
# from matplotlib import image
# from matplotlib import pyplot as plt
# import numpy as np
# import os

# # Read an image file
# path_lena = os.getcwd()
# filename_lena = path_lena + '/samples/' + 'lenna.bmp'
# print("filename:", filename_lena)
# data_lena = image.imread(filename_lena, "w")
# new_img = data_lena.copy()
# size = 200

# # just use another image
# new_img[:size, :size, :] = data_lena[200:400, 200:400, :]
# plt.imshow(new_img)

# import PyPDF2
# file_handle = open("/Users/jiaxu/projects/ROAR-Academy/week_two_excercises/Sense-and-Sensibility-by-Jane-Austen.pdf", "rb") 
# pdfReader = PyPDF2.PdfFileReader(file_handle) 
# page_number = pdfReader.numPages   # this tells you total pages 
# page_object = pdfReader.getPage(0)    # We just get page 0 as example 
# page_text = page_object.extractText()   # this is the str type of full page

# '''
# for loop through each page
#     extract text from every page
# '''
# freq_table = {}
# for i in range(page_number):
#     page_object = pdfReader.getPage(i) # We get page i
#     page_text = page_object.extractText() # this is the str type of full page
#     lines = page_text.split('\n')
    
#     for line in lines:
#         texts = line.split()
#         for word in texts:
#             while len(word) > 0 and not word[-1].isalpha():
#                 word = word[:-1]
#             if word == "CHAPTER" or word == "" or not word.isalpha():
#                 continue
#             if word in freq_table:
#                 freq_table[word] = freq_table[word] + 1
#             else:
#                 freq_table[word] = 1

# print(len(freq_table))

# #lecture 13 excercise 2
# import matplotlib.pyplot as plt

# x = [1,2,3]
# y = [2,4,1]

# plt.plot(x,y)
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.xticks([1,1.5,2,2.5,3])
# plt.title('sample graph')
# plt.show()

# #lecture 13 excercise 1
# #gmt not done
# from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import time

# # Initialization, define some constant
# path = os.path.dirname(os.path.abspath(__file__))
# filename = path + '/airplane.bmp'
# background = plt.imread(filename)

# second_hand_length = 200
# second_hand_width = 2
# minute_hand_length = 150
# minute_hand_width = 6
# hour_hand_length = 100
# hour_hand_width = 10
# center = np.array([256, 256])

# def clock_hand_vector(angle, length):
#     return np.array([length * np.sin(angle), -length * np.cos(angle)])

# # draw an image background
# fig, ax = plt.subplots()
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')
# # ax.spines['left'].set_color('none')
# # ax.spines['bottom'].set_color('none')

# while True:
#     plt.axis('off')
#     plt.imshow(background)

#     gmt = time.gmtime()

#     # First retrieve the time
#     now_time = datetime.now()
    

#     second = now_time.second + now_time.microsecond*10**-6
#     minute = now_time.minute + second/60
#     hour = now_time.hour
#     if hour>12: hour = hour - 12
#     hour += minute/60

#     # Calculate end points of hour, minute, second

#     hour_vector = clock_hand_vector(hour/12*2*np.pi, hour_hand_length)
#     minute_vector = clock_hand_vector(minute/60*2*np.pi, minute_hand_length)
#     second_vector = clock_hand_vector(second/60*2*np.pi, second_hand_length)
#     gmt_vector = clock_hand_vector(hour/60*2*np.pi, hour_hand_length)

#     plt.arrow(center[0], center[1], hour_vector[0], hour_vector[1], head_length = 3, linewidth = hour_hand_width, color = 'yellow')
#     plt.arrow(center[0], center[1], hour_vector[0], hour_vector[1], head_length = 3, linewidth = hour_hand_width, color = 'black')
#     plt.arrow(center[0], center[1], minute_vector[0], minute_vector[1], linewidth = minute_hand_width, color = 'black')
#     plt.arrow(center[0], center[1], second_vector[0], second_vector[1], linewidth = second_hand_width, color = 'red')

#     plt.pause(0.1)
#     plt.clf()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# fig = plt.figure()

# sample_count = 100
# x_sample = 10*np.random.random(sample_count)-5
# y_sample = 2*x_sample - 1 + np.random.normal(0, 1.0, sample_count)

# # plots the parameter space
# ax2 = fig.add_subplot(1,1,1, projection = '3d')

# def penalty(para_a, para_b):
#     global x_sample, y_sample, sample_count

#     squares = (y_sample - para_a*x_sample - para_b)**2
#     return 1/2/sample_count*np.sum(squares)

# a_arr, b_arr = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1) )

# func_value = np.zeros(a_arr.shape)
# for x in range(a_arr.shape[0]):
#     for y in range(a_arr.shape[1]):
#             func_value[x, y] = penalty(a_arr[x, y], b_arr[x, y])

# ax2.plot_surface(a_arr, b_arr, func_value, color = 'red', alpha = 0.8)
# ax2.set_xlabel('a parameter')
# ax2.set_ylabel('b parameter')
# ax2.set_zlabel('f(a, b)')

# # Plot the gradient descent
# def grad(aa):
#     grad_aa = np.zeros(2)
#     update_vector = (y_sample - aa[0] * x_sample - aa[1])
#     grad_aa[0] = - 1/sample_count * x_sample.dot(update_vector)
#     grad_aa[1] = - 1/sample_count * np.sum(update_vector)
#     return grad_aa

# aa = np.array([-4, 4])
# delta = np.inf
# epsilon = 0.001
# learn_rate = 0.001 #changed here
# step_count = 0
# ax2.scatter(aa[0], aa[1], penalty(aa[0],aa[1]), c='b', s=100, marker='*')
# # Update vector aa
# while delta > epsilon:
#     # Gradient Descent
#     aa_next = aa - learn_rate * grad(aa)
#     # Plot the animation
#     ax2.plot([aa[0],aa_next[0]],[aa[1], aa_next[1]],\
#         [penalty(aa[0],aa[1]), penalty(aa_next[0],aa_next[1]) ], 'ko-')
#     delta = np.linalg.norm(aa - aa_next)
#     aa = aa_next
#     step_count +=1
#     fig.canvas.draw_idle()
#     plt.pause(0.1)

# print('Optimal result: ', aa)
# ax2.scatter(aa[0], aa[1], penalty(aa[0],aa[1]), c='b', s=100, marker='*')
# plt.show()
# print('Step Count:', step_count)
## This is course material for Introduction to Modern Artificial Intelligence
## Example code: mlp.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# Load dependencies

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Create data
linearSeparableFlag = False
x_bias = 6

def toy_2D_samples(x_bias ,linearSeparableFlag):
    label1 = np.array([[1, 0]])
    label2 = np.array([[0, 1]])

    if linearSeparableFlag:

        samples1 = np.random.multivariate_normal([5+x_bias, 0], [[1, 0],[0, 1]], 100)
        samples2 = np.random.multivariate_normal([-5+x_bias, 0], [[1, 0],[0, 1]], 100)

        samples = np.concatenate((samples1, samples2 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'rx')
        plt.show()

    else:
        samples1 = np.random.multivariate_normal([5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples2 = np.random.multivariate_normal([-5+x_bias, -5], [[1, 0],[0, 1]], 50)
        samples3 = np.random.multivariate_normal([-5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples4 = np.random.multivariate_normal([5+x_bias, -5], [[1, 0],[0, 1]], 50)

        samples = np.concatenate((samples1, samples2, samples3, samples4 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'bo')
        plt.plot(samples3[:, 0], samples3[:, 1], 'rx')
        plt.plot(samples4[:, 0], samples4[:, 1], 'rx')
        plt.show()

    label1 = np.array([[1, 0]])
    label2 = np.array([[0, 1]])
    labels1 = np.repeat(label1, 100, axis = 0)
    labels2 = np.repeat(label2, 100, axis = 0)
    labels = np.concatenate((labels1, labels2 ), axis =0)
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set

randomOrder = np.random.permutation(200)
trainingX = samples[randomOrder[0:100], :]
trainingY = labels[randomOrder[0:100], :]
testingX = samples[randomOrder[100:200], :]
testingY = labels[randomOrder[100:200], :]

model = Sequential()
#only first layer has input shape
#two layers nextwork
#tone the number of neutrons in each layer, add another layer, different activiation function
model.add(Dense(4, input_shape=(2,), activation='sigmoid', use_bias=True))
# model.add(Dense(4, input_shape=(2,), activation='sigmoid', use_bias=True))
model.add(Dense(4, activation='softmax' ))
model.add(Dense(2, activation='softmax' ))
#model.add(Dense(2, activation='softmax' ))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])

model.fit(trainingX, trainingY, epochs=500, batch_size=10, verbose=1, validation_split=0.2)

# score = model.evaluate(testingX, testingY, verbose=0)
score = 0
for i in range(100):
    predict_x=model.predict(np.array([testingX[i,:]])) 
    estimate=np.argmax(predict_x,axis=1)

    if testingY[i,estimate] == 1:
        score = score  + 1

    if estimate == 0:
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    else: 
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')

print('Test accuracy:', score/100)
plt.show()
