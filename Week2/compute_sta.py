"""
Created on Wed Apr 22 15:21:11 2015

@author: rkp

Code to compute spike-triggered average.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def compute_sta(stim, rho, num_timesteps):
    """Compute the spike-triggered average from a stimulus and spike-train.
    
    Args:
        stim: stimulus time-series
        rho: spike-train time-series
        num_timesteps: how many timesteps to use in STA
        
    Returns:
        spike-triggered average for specified number of timesteps before spike"""
    
    sta = np.zeros((num_timesteps,))

    # This command finds the indices of all of the spikes that occur
    # after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    # Fill in this value. Note that you should not count spikes that occur
    # before 300 ms into the recording.
    num_spikes = len(spike_times)
    print spike_times
    print stim 
    print rho
    print "This is the num of spikes {}".format(num_spikes)
    
    # Compute the spike-triggered average of the spikes found.
    # To do this, compute the average of all of the vectors
    # starting 300 ms (exclusive) before a spike and ending at the time of
    # the event (inclusive). Each of these vectors defines a list of
    # samples that is contained within a window of 300 ms before each
    # spike. The average of these vectors should be completed in an
    # element-wise manner.
    # 
    # Your code goes here.
    for i in range(len(sta)):
        # we're just adding all of the spike vectors 
        # these are the list of samples that are contained within a window of 300ms
        # it's element wise because we're just adding the vector elemnets 
        # then averaging over the nbumber of spikes 
        sta += stim[spike_times[i]-num_timesteps:spike_times[i]]
    return sta/num_spikes # need to average by number of spikes 