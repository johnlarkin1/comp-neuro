'''
Author: John Larkin
Date: 12/27/16
Class: Computational Neuroscience 
Project: Quiz 3 
Purpose: This program is going to help me complete the third quiz about decoding. 
'''

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp

# forgot the square root
def plot_normal(x, mean = 0, sigma = 1):
	return 1.0/np.sqrt(2*np.pi*sigma**2) * np.exp(-((x-mean)**2)/(2*sigma**2))

# found online at:
# http://stackoverflow.com/questions/41368653/intersection-between-gaussian
# thanks to stack overflow for saying that I comment code terribly :(
def solve_gasussians(m1, s1, m2, s2):
    # coefficients of quadratic equation ax^2 + bx + c = 0
    a = (s1**2.0) - (s2**2.0)
    b = 2 * (m1 * s2**2.0 - m2 * s1**2.0)
    c = m2**2.0 * s1**2.0 - m1**2.0 * s2**2.0 - 2 * s1**2.0 * s2**2.0 * np.log(s1/s2)
    x1 = (-b + np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    x2 = (-b - np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    return x1, x2

s1 = np.linspace(0, 10,300)
s2 = np.linspace(0, 14, 300)

y_vals = plot_normal(s1, 5.0, 0.5)
solved_val = solve_gasussians(5.0, 0.5, 7.0, 1.0)
print solved_val
plt.figure('Baseline Distributions')
plt.title('Baseline Distributions')
plt.xlabel('Response Rate')
plt.ylabel('Probability')
plt.plot(s1, plot_normal(s1, 5.0, 0.5),'r', label='s1')
plt.plot(s2, plot_normal(s2, 7.0, 1.0),'b', label='s2')
for x_pt in solved_val:
	plt.plot(x_pt, plot_normal(x_pt, 7.0, 1.0), 'mo')
plt.legend()
plt.show()

# question is:
# Which of these firing rates would make the best decision threshold for us
# in determining the value of s given a neuron's firing rate?
# Also recall that we want to weight it so that it is TWICE as bad to think it is
# s2 rather than s1

# They only give us four options. Let's just compute the pdf at some given value
