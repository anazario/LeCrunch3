import h5py
import numpy as np
import matplotlib.pyplot as plt
import CFD

if __name__ == '__main__':
    import optparse
    import sys
    import pulse

    usage = "usage: %prog [-n]"
    parser = optparse.OptionParser(usage, version="%prog 0.1.0")
    parser.add_option("-i", type=str, dest="infile",
                      help="input file name", default="output.h5")
    parser.add_option("--ref", type=int, dest="ref_ch",
                      help="channel that is the reference sensor", default=1)
    parser.add_option("--ch", type=int, dest="channels",
                      help="amount of channels saved in input file", default=1)
    parser.add_option("--cfd", type=float, dest="cfd",
                      help="percentage for CFD time estimation", default=0.3)
    parser.add_option("--thresh", type=float, dest="threshold",
                      help="sample amplitude threshold value", default=0.025)
    parser.add_option("--baseline", type=float, dest="baseline",
                      help="baseline time in nanoseconds", default=20)
    parser.add_option("--plot", action="store_true", dest="isPlot",
                      help="if called, save the time difference plot", default=False)            
    (options, args) = parser.parse_args()

    if(options.channels < 1 or options.channels > 8):
        sys.exit(f'{options.channels} is not a valid amount of channels!')

    if(options.cfd < 0.01 or options.cfd > 1.0):
        sys.exit("CFD percentage must be a number between 0 and 1!")
        
    f = h5py.File(options.infile,'r')

    num_triggers = len(f['ch1_samples'])
    num_samples = len(f['ch1_samples'][0])
    time_begin = f['ch1_trig_offset'][0]*1e9
    time_end = f['ch1_horiz_scale'][0]*(num_samples-1)*1e9+time_begin
    time = np.linspace(time_begin, time_end, num_samples)
    sampling_freq = num_samples/(time_end-time_begin)

    amp_max = []
    cfd_times = []
    
    for channel in range(1, options.channels+1):
        temp_cfd = []
        temp_amp_max = []
        samples = f[f'ch{channel}_samples']
        horiz_scale = f[f'ch{channel}_horiz_scale']
        trig_offset = f[f'ch{channel}_trig_offset']
        for trig in range(num_triggers):
            sample = -1e3*(samples[trig]-np.max(samples[trig]))
            temp_amp_max.append(np.max(sample))
            sample = sample/np.max(sample)
            cfd_val = CFD.cfd(sample,options.threshold,int(sampling_freq*options.baseline),options.cfd)
            if cfd_val is not None:
                temp_cfd.append((cfd_val*horiz_scale[trig]+trig_offset[trig])*1e9)
            else:
                temp_cfd.append(-999.)
        cfd_times.append(temp_cfd)
        amp_max.append(temp_amp_max)
    
    time_diff = []
    time_diff_channels = []
        
    for channel in range(1,options.channels+1):
        temp = []
        
        if(channel is not options.ref_ch):
            time_diff_arr = np.subtract(np.array(cfd_times[options.ref_ch-1]),np.array(cfd_times[channel-1]))
            for trig in range(num_triggers):
                diff = time_diff_arr[trig]
                maximum = amp_max[channel-1][trig]
                if diff > -3 and diff < 0 and maximum > 200 and maximum < 700:
                    temp.append(diff)
            time_diff.append(temp)
            time_diff_channels.append(channel)

    for data in time_diff:
        print(f'Channel {time_diff_channels[time_diff.index(data)]}:')
        
        from iminuit import Minuit
        from probfit import UnbinnedLH, gaussian

        unbinned_likelihood = UnbinnedLH(gaussian, np.array(data))
        minuit = Minuit(unbinned_likelihood, mean=0.1, sigma=1.1)
        minuit.migrad()
        parameters = minuit.values
        errors = minuit.errors
        mean = parameters['mean']
        sigma = parameters['sigma']
        mean_err = errors['mean']
        sigma_err = errors['sigma']
        print(f'Mean: {mean:.3} +/- {mean_err:.3} ns')
        print(f'Standard Dev: {sigma:.3} +/- {sigma_err:.3} ns')
        
        n_bin = 33
        fig, ax = plt.subplots()
        ax.set_title(f'Time Difference Distribution (CFD {int(options.cfd*100)}%)')
        ax.set_xlabel('Time Difference [ns]')
        ax.set_ylabel('Triggers')
        unbinned_likelihood.draw(minuit, bins=n_bin, show_errbars=None)
        plt.hist(data, n_bin) 
        plt.savefig(f'ch{time_diff_channels[time_diff.index(data)]}_time_diff_hist_CDF{int(options.cfd*100)}.pdf')
