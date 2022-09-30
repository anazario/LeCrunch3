import h5py
import math
import numpy as np
import matplotlib.pyplot as plt

from InvKLJfilter import *

def PlotLeadingEigVectors(input_file, max_idx = 5):
    colors = ['black','red', 'blue', 'green', 'orange']
    f = h5py.File(input_file,'r')
    channels = [1,2,3,4]

    xaxis = np.linspace(0,126,127)
    for channel in channels:
        fig, ax = plt.subplots()
        ax.set_xlabel('Nth Sample')
        ax.set_ylabel('Eigenvector Sample Value')
        ax.set_title(f'Channel {channel}')
    
        samples = f[f'ch{channel}_samples']
        num_samples = len(f[f'ch{channel}_samples'][0])
        
        eig_values, eig_vectors = GetBasisInvKLJ(samples)
        for idx in range(max_idx):
            temp_eig_vector = eig_vectors[:,idx]
            plt.plot(xaxis, temp_eig_vector, color=colors[idx], label=f'λ = {round(eig_values[idx],4)}')    

        plt.legend()    
        plt.savefig(f"channel{channel}_{max_idx}_leading_eigenvectors.pdf") 
        plt.clf()

def PlotWaveform(input_file, channel, pulse_index, useFilter = False):

    f = h5py.File(input_file,'r')

    samples = f[f'ch{channel}_samples']
    num_samples = len(f[f'ch{channel}_samples'][pulse_index])

    time_begin = f[f'ch{channel}_trig_offset'][pulse_index]*1e9
    time_end = f[f'ch{channel}_horiz_scale'][pulse_index]*(num_samples-1)*1e9+time_begin
    time = np.linspace(time_begin, time_end, num_samples)
    
    waveform = samples[:][pulse_index]
    waveform = np.mean(waveform[:int(num_samples/3)]) - waveform

    if(useFilter):
        filter_invKLJ = KLJ_filter(samples, 5)
        waveform = np.matmul(filter_invKLJ, waveform)
    
    fig, ax = plt.subplots()
    ax.set_ylabel('Amplitude [V]')
    ax.set_xlabel('Time [ns]')
    ax.set_title(f'Channel {channel}: Pulse {pulse_index}')
    plt.plot(time, waveform)

    label = 'unfiltered'
    if useFilter:
        label = 'filtered'

    plt.savefig(f'channel{channel}_pulse{pulse_index}_{label}.pdf')

if __name__ == '__main__':

    input_file = "source_coincidence_test9.h5"

    PlotWaveform(input_file, 1, 1, True)
    PlotLeadingEigVectors(input_file, 3)
