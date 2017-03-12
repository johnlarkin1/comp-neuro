# Author: John Larkin
# Date: 3/9/17 
# Project:
# 	Taking this Computational Neuroscience class so I can talk to my sister about some of these things 
# 	This is Week 7 quiz

import numpy as np 
import pickle
import matplotlib.pyplot as plt

USE_PLOTS = False

Q = np.matrix([[0.15, 0.1],
	           [0.1 , 0.12]])
print("This is Q:",Q)
eigenvals, eigenvecs = np.linalg.eig(Q)

for eigenvec in eigenvecs:
	print("This is the eigenvec: {}".format(eigenvec))

# Then we need to look for any of the possible solutions that is proportional to an eigenvalue
# Only one existed for the Coursera quiz

# Going to implement Oja's Rule 
# discrete wise:
# delta w = delta t * 1/ tau_w * (vu- alpha v^2 w)

with open('c10p1.pickle', 'rb') as f:
    data = pickle.load(f)
    data = data['c10p1']
# Let's see what form it's in
print("This is our data: {}".format(data))

if USE_PLOTS:
	plt.figure()
	plt.title('Imported 2D Neuron Data')
	plt.xlabel('X Coordinates')
	plt.ylabel('Y Coordinates')
	plt.plot(data[:,0], 'ro')
	plt.plot(data[:,1], 'bo')
	plt.show()

# Let's get the mean of the x coordinates and the mean of the y coordinates
xavg, yavg = np.mean(data[:,0]), np.mean(data[:,1])

# Let's subtract things off to have it be zero averaged
new_xdata, new_ydata = data[:,0]-xavg, data[:,1]-yavg

# Let's form our new data
new_data = np.column_stack((new_xdata, new_ydata))

# Let's check to make sure
if USE_PLOTS:
	plt.figure()
	plt.title('Imported 2D Neuron Data')
	plt.xlabel('X Coordinates')
	plt.ylabel('Y Coordinates')
	plt.plot(new_data[:,0], 'ro')
	plt.plot(new_data[:,1], 'bo')
	plt.show()


eta = 1
alpha = 1
step_t = 0.01
# Going to prematurely detect convergence
'''
Start with a random vector w_0 
Feed in a data point and update 
If you hit the last data point, then go back to the first one and repeat
'''

w_0 = np.random.rand(2)
print("This is w0: {}".format(w_0))
w_i = w_0
for i in range(100000):
	u = new_data[i % len(new_data), :]
	v = np.dot(u, w_i)
	w_next = w_i + step_t * eta * (v * u - alpha * v**2 * w_i)
	w_i = w_next
print("This is our w_final: {}".format(w_i))

# Ah so w_i flips in between two different eigenvectors! You might have to run the code like 5 or 10 times, but you should see the results
# Let's estimate our covariance matrix

corr_matrix = np.corrcoef(new_xdata, new_ydata)
cov_matrix = np.cov(new_xdata, new_ydata)

corr2_matrix = np.dot(new_data.T, new_data)/len(new_data)
# Because the data is mean centered the correlation matrix will be the same as the covariance matrix
# Let's check this assertion
# print(corr_matrix)
# print(cov_matrix)
# print(corr2_matrix)

# All the same by a scalar factor
eigenvals, eigenvecs = np.linalg.eig(corr_matrix)

for eigenvec in eigenvecs:
	print("This is the eigenvec: {}".format(eigenvec))



