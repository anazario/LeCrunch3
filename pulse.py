import CFD
import numpy as np
import matplotlib.pyplot as plt
import sys

class Pulse:

    def __init__(self, voltage, time_start, time_end):
        self.voltage = voltage
        self.num_samples = len(voltage)
        self.time_start = time_start
        self.time_end = time_end
        self.sampling_frequency = self.num_samples/(time_end-time_start)
        self.noise_rms = np.std(voltage[:int(0.3*self.num_samples)])

    def normalize(self):
        self.voltage /= np.max(self.voltage)

    def flip_sign(self):
        self.voltage *= -1

    def get_num_samples(self):
        return len(self.voltage)

    def get_time_array(self):
        return np.linspace(self.time_start, self.time_end, self.num_samples)

    def get_max_amplitude(self):
        amplitude = -1*self.voltage
        return np.max(amplitude)

    def save_plot(self, name, flipSign=False, isNormal=False):
        sample = self.voltage
        if flipSign:
            self.flip_sign()
        if isNormal:
            self.normalize()
            
        fig, ax = plt.subplots()
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Amplitude [mV]')
        plt.plot(self.get_time_array(), self.voltage)
        plt.savefig(name+".pdf")
        
    def get_cfd_index(self, fraction=0.3, threshold=None, baseline=None):
        if threshold is None:
            threshold=(self.noise_rms)*6
        if baseline is None:
            baseline=int(0.3*self.num_samples)
        return CFD.cfd(self.voltage, threshold, int(self.sampling_frequency*base_line), fraction)


def get_pulse(h5_file, channel, pulse_index):

    import h5py
    
    file_h5 = h5py.File(h5_file, 'r')

    samples      = file_h5[f'ch{channel}_samples']
    horiz_offset = file_h5[f'ch{channel}_horiz_offset']
    horiz_scale  = file_h5[f'ch{channel}_horiz_scale']
    trig_time    = file_h5[f'ch{channel}_trig_time']
    trig_offset  = file_h5[f'ch{channel}_trig_offset']

    n_samples   = (len(samples))
    sample_size = (len(samples[0]))

    if (pulse_index > n_samples):
        sys.exit("Specified index is out of bounds!")

    time_begin = trig_offset[pulse_index]*1e9
    time_end   = (horiz_scale[pulse_index]*(sample_size-1)-trig_offset[pulse_index])*1e9
    time       = np.linspace(time_begin, time_end, sample_size)

    pulse = samples[pulse_index]

    return pulse, time
'''
import h5py

f = h5py.File('output.h5', 'r')
samples = f['ch1_samples']
time_begin = f['ch1_trig_offset'][0]*1e9
num_samples = len(samples[0])
time_end = f['ch1_horiz_scale'][0]*(num_samples-1)*1e9+time_begin


pulse_ch1 = Pulse(samples[0]*1e3, time_begin, time_end)
print(pulse_ch1.get_cfd_index(0.4))
#pulse_ch1.save_plot("test_plot", True, True)
'''
