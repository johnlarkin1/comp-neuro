# Computational Neuroscience 
# Author: John Larkin
# Date: 12/29/16
# Project: 
# Trying to solve quiz 4 for Coursera's Comptuational Neuroscience class. 

import numpy as np 
import matplotlib.pyplot as plt
import pickle

# Problem 1. 
def problem1():
    '''
    Suppose that we have a neuron which, in a given time period, will fire with probability 0.1, yielding a Bernoulli distribution for the neuron's firing (denoted by the random variable F = 0 or 1) with P(F = 1) = 0.1.

    Which of these is closest to the entropy H(F) of this distribution (calculated in bits, i.e., using the base 2 logarithm)?
    '''
    # H(x) = \sum_i to # options (p(x)) \log (p(x))
    return - (.1 * np.log2(.1) + .9 * np.log2(.9))

def problem2():
    '''
    Continued from Question 1:

    Now lets add a stimulus to the picture. Suppose that we think this neuron's activity is related to a light flashing in the eye. Let us say that the light is flashing in a given time period with probability 0.10. Call this stimulus random variable S.

    If there is a flash, the neuron will fire with probability 1/2. If there is not a flash, the neuron will fire with probability 1/18. Call this random variable F (whether the neuron fires or not).

    Which of these is closest, in bits (log base 2 units), to the mutual information MI(S,F)?
    '''
    # Well 
    # MI(S,R) = total entropy - average noise entropy
    # MI(S,R) = - total entropy (part 1) - sum over the stimuli [ p(s) [ - sum over the responses [p(r|s) * log (p(r|s))]]]
    probOfLightFlashing = 0.10
    # this is rando var S
    # IF there is a flash (conditional probability), the neuron will fire with prob 1/2. IF NOT a flash, neuron will fire with prob 1/18
    # Call this rando of F
    # So really MI (S , F)
    # just one stimulu
    entropyOfFlash = - 0.10 * ( 0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
    entropyOfNoFlash = - 0.90 * (1/18.0 * np.log2 ( 1/18.0) + 17/18.0 * np.log2(17/18.0))
    return problem1() - (entropyOfFlash + entropyOfNoFlash)

def problem3and4and5():
    '''
    This math from lecture 4.3 could potentially be intimidating, but in fact the concept is really simple. Getting an intuition for it will help with many types of problems. Let's work out a metaphor to understand it.

    Suppose we want to build a complex image. We could do that by layering a whole bunch of pieces together (mathematically - summing). This is like drawing on transparencies with various levels of opacity and putting them on top of each other. Those familiar with Photoshop or Gimp will recognize that concept. If we had to build an image in Photoshop with a bicycle on a road, for instance, perhaps we could have an image of a sky, and one of the road, and one of the bike. We could "add" these pieces together to make our target image.

    Of course, if our neural system was trying to make visual fields that worked for any sort of input, we would want more than just roads, skies, and bikes to work with! One possibility is to have a bunch of generic shapes of various sizes, orientations, and locations within the image. If we chose the right variety, we could blend/sum these primitive pieces together to make just about any image! One way to blend them is to let them have varying transparencies/opacities, and to set them on top of each other. That is what we would call a weighted sum, where the weights are how transparent each piece is.

    Of course, we may not want to have too many possible shapes to use. As mentioned in the video, the organism likely wants to conserve energy. That means having as few neurons firing as possible at once. If we conceptually make a correlation between these shapes and the neurons, then we can point out we would want to use as few shapes as we could while maintaining an accurate image.

    This math gives us a way of summing a bunch of pieces together to represent an image, to attempt to make that representation look as much like the image as possible, and to make that representation efficient - using as few pieces as possible. That is a lot of work for two lines of math!

    Now let's put this metaphor into action to understand what all these symbols mean. I'll give you one to start with. The vector x in the equation above represents the coordinates of a point in the image. Now you fill in the rest:

    What do the phi_i's, called the "basis functions," represent in our metaphor?
    '''

    print("Question 3:")
    print("This phi's are the pieces that make up the image")
    print("Question 4:")
    print("The epsilon's are the difference between the actual image and the representation")
    print("Question 5:")
    print("The a's are the level of transparency vs opacity/influence of each piece")
    print("Question 6:")
    print("The lambda represents the importance of coding effiency")

def problem7(stim, neuron1, neuron2, neuron3, neuron4):
    '''
    This exercise is based on a set of artificial "experiments" that we've run on four simulated neurons that emulate the behavior found in the cercal organs of a cricket. Please note that all the supplied data is synthetic. Any resemblance to a real cricket is purely coincidental.

    In the first set of experiments, we probed each neuron with a range of air velocity stimuli of uniform intensity and differing direction. We recorded the firing rate of each of the neurons in response to each of the stimulus values. Each of these recordings lasted 10 seconds and we repeated this process 100 times for each neuron-stimulus combination.

    We've supplied you with a .mat file for each of the neurons that contains the recorded firing rates (in Hz). These are named neuron1, neuron2, neuron3, and neuron4. The stimulus, that is, the direction of the air velocity, is in the vector named stim.
    '''
    meanFiringRateNeuron1 = np.mean(neuron1, axis =0)
    meanFiringRateNeuron2 = np.mean(neuron2, axis =0)
    meanFiringRateNeuron3 = np.mean(neuron3, axis =0)
    meanFiringRateNeuron4 = np.mean(neuron4, axis =0)
    plt.plot(stim, meanFiringRateNeuron1, 'r-', label='Neuron 1')
    plt.plot(stim, meanFiringRateNeuron2, 'b-', label='Neuron 2')
    plt.plot(stim, meanFiringRateNeuron3, 'm-', label='Neuron 3')
    plt.plot(stim, meanFiringRateNeuron4, 'k-', label='Neuron 4')
    plt.title('Tuning Curve for Imaginary Cricket')
    plt.xlabel('Stimulus')
    plt.ylabel('Mean Firing Rate of Neuron')
    plt.legend(prop={'size':10})
    plt.show()

def problem8(stim, neuron1, neuron2, neuron3, neuron4):
    '''
    We have reason to suspect that one of the neurons is not like the others. Three of the neurons are Poisson neurons (they are accurately modeling using a Poisson process), but we believe that the remaining one might not be.

    Which of the neurons (if any) is NOT Poisson?
    
    Hint: Think carefully about what it means for a neuron to be Poisson. You may find it useful to review the last lecture of week 2. Note that we give you the firing rate of each of the neurons, not the spike count. You may find it useful to convert the firing rates to spike counts in order to test for "Poisson-ness", however this is not necessary.
    
    In order to realize why this might be helpful, consider the fact that, for a constant a and a random variable X. What might this imply about the Poisson statistics (like the Fano factor) when we convert the spike counts (the raw output of the Poisson spike generator) into a firing rate (what we gave you)?
    '''
    # Specified, that they divided each recoding, which was 10 seconds, by 10
    # so mean should be scale dby 1/10 as noted by hint
    # variance scaled by 1/100
    # so then just need to find the variance and means
    expectedFano = (1/10.0)/(1/100.0)
    print("This expected Fano Factor: {}".format(expectedFano))
    meanFiringRateNeuron1 = np.mean(neuron1, axis =0)
    varFiringRateNeuron1 = np.var(neuron1, axis=0)
    meanFiringRateNeuron2 = np.mean(neuron2, axis =0)
    varFiringRateNeuron2 = np.var(neuron2, axis=0)
    meanFiringRateNeuron3 = np.mean(neuron3, axis =0)
    varFiringRateNeuron3 = np.var(neuron3, axis=0)
    meanFiringRateNeuron4 = np.mean(neuron4, axis =0)
    varFiringRateNeuron4 = np.var(neuron4, axis=0)

    validIdxN1 = meanFiringRateNeuron1.nonzero()[0]
    validIdxN2 = meanFiringRateNeuron2.nonzero()[0]
    validIdxN3 = meanFiringRateNeuron3.nonzero()[0]
    validIdxN4 = meanFiringRateNeuron4.nonzero()[0]

    fanN1 = meanFiringRateNeuron1[validIdxN1] / varFiringRateNeuron1[validIdxN1]
    fanN2 = meanFiringRateNeuron2[validIdxN2] / varFiringRateNeuron2[validIdxN2]
    fanN3 = meanFiringRateNeuron3[validIdxN3] / varFiringRateNeuron3[validIdxN3]
    fanN4 = meanFiringRateNeuron4[validIdxN4] / varFiringRateNeuron4[validIdxN4]

    print("The avg fano factor for neuron 1 are: {}".format(np.mean(fanN1))) 
    print("The avg fano factor for neuron 2 are: {}".format(np.mean(fanN2))) 
    print("The avg fano factor for neuron 3 are: {}".format(np.mean(fanN3))) 
    print("The avg fano factor for neuron 4 are: {}".format(np.mean(fanN4)))

    print("Neuron 3 is obviously not a Poisson neuron")
    return (np.max(meanFiringRateNeuron1), np.max(meanFiringRateNeuron2), np.max(meanFiringRateNeuron3), np.max(meanFiringRateNeuron4))
    

def problem9(maxes):
    '''
    Finally, we ran an additional set of experiments in which we exposed each of the neurons to a single stimulus of unknown direction for 10 trials of 10 seconds each. We have placed the results of this experiment in the following file:
    '''
    with open('pop_coding_2.7.pickle', 'rb') as f:
        data = pickle.load(f)
    print data.keys()
    '''
    pop_coding contains four vectors named r1, r2, r3, and r4 that contain the responses (firing rate in Hz) of the four neurons to this mystery stimulus. It also contains four vectors named c1, c2, c3, and c4. These are the basis vectors corresponding to neuron 1, neuron 2, neuron 3, and neuron 4.

    Decode the neural responses and recover the mystery stimulus vector by computing the population vector for these neurons. You should use the maximum average firing rate (over any of the stimulus values in 'tuning.mat') for a neuron as the value of rmax for that neuron. That is, rmax should be the maximum value in the tuning curve for that neuron.'
    '''
    r1 = data['r1']
    r2 = data['r2']
    r3 = data['r3']
    r4 = data['r4']
    c1 = data['c1']
    c2 = data['c2']
    c3 = data['c3']
    c4 = data['c4']
    '''
    In neuroscience, a population vector is the sum of the preferred directions of a population of neurons, weighted by the respective spike counts.

    The formula for computing the (normalized) population vector, {\displaystyle F} F, takes the following form:
    sum from 1 to 4 of r / r_max * c_a
    '''
    info = zip([r1, r2, r3, r4], [c1, c2, c3, c4], [maxes[0], maxes[1], maxes[2], maxes[3]])
    population_vector = np.zeros(len(c1))
    for (r, c, rmax) in info:
        population_vector += np.mean(r) /rmax * c 
    angle = np.arctan(-population_vector[0]/population_vector[1])
    answer = 180 - angle * 180.0/ np.pi
    print("The direction of the population vector is: {}".format(answer))
    return answer

if __name__ == '__main__':
    ans1 = problem1()
    print("Question 1:")
    print("The solution to problem 1 is: {}".format(ans1))
    ans2 = problem2()
    print("Question 2:")
    print("The solution to problem 2 is: {}".format(ans2))
    problem3and4and5()
    with open('tuning_2.7.pickle', 'rb') as f:
        data = pickle.load(f)
    stim = data['stim']
    neuron1 = data['neuron1']
    neuron2 = data['neuron2']
    neuron3 = data['neuron3']
    neuron4 = data['neuron4']
    print("Question 7:")
    problem7(stim, neuron1, neuron2, neuron3, neuron4)
    print("Question 8:")
    maxes = problem8(stim, neuron1, neuron2, neuron3, neuron4)
    problem9(maxes)




