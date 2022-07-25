#!/usr/bin/env python3
# LeCrunch3
# Copyright (C) 2021 Nicola Minafra
#
# based on
#
# LeCrunch2
# Copyright (C) 2014 Benjamin Land
#
# based on
#
# LeCrunch
# Copyright (C) 2010 Anthony LaTorre 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import time
import string
import struct
import socket
import h5py
import numpy as np
import CFD
from LeCrunch3 import LeCrunch3

def subtract_nan(list1,list2):
    list1 = list(list1)
    list2 = list(list2)

    if len(list1) != len(list2):
        sys.exit("Both lists must have equal dimensions!")

    length = len(list1)

    output_list = []
    
    for index in range(length):
        if np.isnan(list1[index]) or np.isnan(list2[index]):
            pass
        else:
            output_list.append(list1[index]-list2[index])

    return output_list
    
def fetchAndSaveFast(filename, nevents, nsequence, ip, reference_ch, timeout=1000):
    '''
    Fetch and save waveform traces from the oscilloscope 
    with ADC values and with all info needed to reconstruct the waveforms
    It is faster than fetchAndSaveSimple but it requires a bit more code to analyze the files
    '''
    scope = LeCrunch3(ip, timeout=timeout)
    scope.clear()
    scope.set_sequence_mode(nsequence)
    channels = scope.get_channels()
    settings = scope.get_settings()

    if b'ON' in settings['SEQUENCE']:
        sequence_count = int(settings['SEQUENCE'].split(b',')[1])
    else:
        sequence_count = 1
    print(sequence_count)
        
    if nsequence != sequence_count:
        print('Could not configure sequence mode properly')
    if sequence_count != 1:
        print(f'Using sequence mode with {sequence_count} traces per aquisition')
    
    rms = {}
    minimum = {}
    maximum = {}
    rate = {}
    cfd_times = {}
    time_diff = {}

    f = h5py.File(filename, 'w')
    for command, setting in settings.items():
        f.attrs[command] = setting
    current_dim = {}
    
    print("Channels: ", channels)
    for channel in channels:
        wave_desc = scope.get_wavedesc(channel)
        current_dim[channel] = wave_desc['wave_array_count']//sequence_count
        f.create_dataset(f'ch{channel}_samples', (nevents,current_dim[channel]), dtype='f8')
        ## Save attributes in the file
        for key, value in wave_desc.items():
            try:
                f[f"ch{channel}_samples"].attrs[key] = value
            except ValueError:
                pass
        f.create_dataset(f'ch{channel}_horiz_offset', (nevents,), dtype='f8')
        f.create_dataset(f'ch{channel}_horiz_scale', (nevents,), dtype='f8')
        f.create_dataset(f'ch{channel}_trig_offset', (nevents,), dtype='f8')
        f.create_dataset(f'ch{channel}_trig_time', (nevents,), dtype='f8')
        f.create_dataset(f'ch{channel}_cfd_times', (nevents,), dtype='f8')
        
        # Add here all measurements you want...                                                                                   
        minimum[channel] = []
        maximum[channel] = []
        rms[channel] = []
        rate[channel] = []
        cfd_times[channel] = []
        time_diff[channel] = []

    try:
        i = 0
        while i < nevents:
            print(f'\rfetching event: {i}')
            sys.stdout.flush()
            try:
                scope.trigger()
                for channel in channels:
                    wave_desc, trg_times, trg_offsets, wave_array = scope.get_waveform_all(channel)
                    num_samples = wave_desc['wave_array_count']//sequence_count
                    filt_trg_times = []
                    if current_dim[channel] < num_samples:
                        current_dim[channel] = num_samples
                        f[f'ch{channel}_samples'].resize(current_dim[channel],1)
                    traces = wave_array.reshape(sequence_count, wave_array.size//sequence_count)
                    #necessary because h5py does not like indexing and this is the fastest (and man is it slow) way
                    scratch = np.zeros((current_dim[channel],),dtype=wave_array.dtype)
                    for n in range(0,sequence_count):
                        scratch[0:num_samples] = traces[n] 
                        
                        samples = -wave_desc['vertical_offset'] + scratch*wave_desc['vertical_gain']

                        time_begin = trg_offsets[n]*1e9
                        time_end = (wave_desc['horiz_interval']*(num_samples-1)-trg_offsets[n])*1e9
                        time = time_end - time_begin
                        sample_freq = num_samples/(time)
                        horiz_scale = wave_desc['horiz_interval']

                        if np.abs(np.max(samples) - np.min(samples)) > 12*np.std(samples[:100]):
                            filt_trg_times.append(trg_times[n])
                        rms[channel].append(np.std(samples))
                        minimum[channel].append(np.min(samples))
                        maximum[channel].append(np.max(samples))

                        f[f'ch{channel}_samples'][i+n] = samples                                           
                        f[f'ch{channel}_horiz_offset'][i+n] = wave_desc['horiz_offset']
                        f[f'ch{channel}_horiz_scale'][i+n] = wave_desc['horiz_interval']
                        f[f'ch{channel}_trig_offset'][i+n] = trg_offsets[n]
                        f[f'ch{channel}_trig_time'][i+n] = trg_times[n]

                        try:
                            if minimum[channel][n] < -0.025:
                                cfd_times[channel].append(CFD.cfd(samples,0.001,int(20*sample_freq))*horiz_scale+trg_offsets[n])
                            else:
                                cfd_times[channel].append(None)
                        except:
                            pass        
                        f[f'ch{channel}_cfd_times'][i+n] = cfd_times[channel][n]

                    rate[channel].append(1.0/np.mean(np.diff(trg_times)))
                    
            except Exception as e:
                print('Error\n' + str(e))
                scope.clear()
                continue
            i += sequence_count
    except KeyboardInterrupt:
        print('\rUser interrupted fetch early')
    finally:
        #Time difference                                                                                                                                      
        ref_time = np.array(cfd_times[reference_ch])

        for channel in channels:
            if(channel is not reference_ch):
                temp_time_diff = []
                for n in range(nevents):
                    dut_time = np.array(cfd_times[channel])
                    time_diff[channel] = np.subtract(dut_time, ref_time)
                    f.create_dataset(f'ch{channel}_time_diff', data = time_diff[channel])
            else:
                time_diff[channel] = np.zeros(len(cfd_times[channel]))
        print('\r', )
        scope.clear()

        for channel in channels:
            if len(maximum[channel])>1:
                print(f'Channel {channel}:')
                print(f'\tRMS: {np.mean(rms[channel])*1000:.3} mV')
                print(f'\tMinimum Mean: {np.mean(minimum[channel])*1000:.3} mV')
                print(f'\tMaximum Mean: {np.mean(maximum[channel])*1000:.3} mV')
                print(f'\tAvg rate: {np.mean(rate[channel]):.3e} Hz')
                if(channel is not reference_ch):
                    print(f'\tTime Difference Mean: {np.mean(time_diff[channel])*1e9:.3} ns')
                    print(f'\tTime Difference RMS: {np.std(time_diff[channel])*1e9:.3} ns')
                print('')
            else:
                print(f'Channel {channel}:')
                print(f'\tRMS: {rms[channel][0]*1000:.3} mV')
                print(f'\tMinimum Mean: {minimum[channel][0]*1000:.3} mV')
                print(f'\tMaximum Mean: {maximum[channel][0]*1000:.3} mV')
                print(f'\tAvg rate: {rate[channel][0]:.3e} Hz')
                if(channel is not reference_ch):
                    print(f'\tTime Difference Mean: {time_diff[channel][0]*1e9:.3} ns')
                print('')
        print('\r', )
        f.close()
        scope.clear()
        return i

if __name__ == '__main__':
    import optparse

    usage = "usage: %prog <filename/prefix> [-n] [-s]"
    parser = optparse.OptionParser(usage, version="%prog 0.1.0")
    parser.add_option("-i", type="str", dest="ip",
                      help="IP address of the scope", default="127.0.0.1")
    parser.add_option("-n", type="int", dest="nevents",
                      help="number of events to capture in total", default=1000)
    parser.add_option("-s", type="int", dest="nsequence",
                      help="number of sequential events to capture at a time", default=1)
    parser.add_option("--time", action="store_true", dest="time",
                      help="append time string to filename", default=False)
    parser.add_option("-r", type="int", dest="reference",
                      help="reference sensor channel", default=1)                  
    (options, args) = parser.parse_args()

    if len(args) < 1:
        sys.exit(parser.format_help())
    
    if options.nevents < 1 or options.nsequence < 1:
        sys.exit("Arguments to -s or -n must be positive")
    
    filename = args[0] + time.strftime("_%d_%b_%Y_%H:%M:%S", time.localtime()) + '.h5' if options.time else args[0] + '.h5'
    print(f'Saving to file {filename}')

    start = time.time()
    count = fetchAndSaveFast(filename, options.nevents, options.nsequence, options.ip, options.reference)
    elapsed = time.time() - start
    if count > 0:
        print(f'Completed {count} events in {elapsed:.3f} seconds.')
        print(f'Averaged {elapsed/count:.5f} seconds per acquisition.')
