import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import optparse
    import sys
    import CFD

    usage = "usage: %prog [-n]"
    parser = optparse.OptionParser(usage, version="%prog 0.1.0")
    parser.add_option("-i", type=str, dest="infile",
                      help="input file name", default="output.h5")
    parser.add_option("--ref", type=int, dest="ref_ch",
                      help="channel that is the reference sensor", default=1)
    parser.add_option("--ch", type=int, dest="channels",
                      help="amount of channels saved in input file", default=1)
    parser.add_option("--plot", action="store_true", dest="isPlot",
                      help="if called, save the time difference plot", default=False)            
    (options, args) = parser.parse_args()

    if(options.channels < 1 or options.channels > 8):
        sys.exit(f'{options.channels} is not a valid amount of channels!')

    f = h5py.File(options.infile,'r')

    num_triggers = len(f['ch1_samples'])
    num_samples = len(f['ch1_samples'][0])
    time_begin = f['ch1_trig_offset'][0]*1e9
    time_end = f['ch1_horiz_scale'][0]*(num_samples-1)*1e9+time_begin
    time = np.linspace(time_begin, time_end, num_samples)
    sampling_freq = num_samples/(time_end-time_begin)

    amplitude = []

    for channel in range(1, options.channels+1):
        temp = []
        samples = f[f'ch{channel}_samples']
        horiz_scale = f[f'ch{channel}_horiz_scale']
        trig_offset = f[f'ch{channel}_trig_offset']
        for trig in range(num_triggers):
            sample = -1*(samples[trig]-np.max(samples[trig]))
            maximum = np.max(sample)*1e3
            if maximum > 150:
                temp.append(maximum)
        amplitude.append(temp)
        
    for amp in amplitude:
        fig = plt.figure()
        #plt.yscale('log')
        plt.hist(amp,40)
        plt.savefig(f'ch{amplitude.index(amp)+1}_amplitude.pdf')
        plt.clf()
