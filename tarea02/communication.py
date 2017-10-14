
import numpy as np
import pylab as pl
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample
import random

class Communication():

    @staticmethod
    def mod_fsk(x, Fs, Fbit, Fc, Fdev):
        # total number of bits
        n = len(x)

        # one second of data
        t = np.arange(0, float(n) / float(Fbit), 1 / float(Fs), dtype=np.float)

        # extend the data_in to account for the bitrate and convert 0/1 to frequency
        m = np.zeros(0).astype(float)
        # print(Fc-Fdev)
        for bit in x:
            if bit == 0:
                m = np.hstack((m, np.multiply(np.ones(int(Fs / Fbit)), Fc + Fdev).astype(float)))
            else:
                m = np.hstack((m, np.multiply(np.ones(int(Fs / Fbit)), Fc - Fdev).astype(float)))

        # #calculate the output of the VCO
        y = np.zeros(0)
        y = 1.0 * np.cos(2 * np.pi * np.multiply(m, t))
        return y

    @staticmethod
    def noise_channel(x, A_n):
        #create some noise
        noise = (np.random.randn(len(x))+1)*A_n
        snr = 10*np.log10(np.mean(np.square(x)) / np.mean(np.square(noise)))
#        print("SNR = %fdB" % snr)
        y=np.add(x,noise)
        return y, snr

    @staticmethod
    def demod_fsk(x, Fs, Fbit, Fc, Fdev):
        # deffierentiator
        y_diff = np.diff(x,1)

        # create an envelope detector and then low-pass filter
        y_env = np.abs(sigtool.hilbert(y_diff))
        h=signal.firwin( numtaps=100, cutoff=Fbit*2, nyq=Fs/2)
        y_filtered=signal.lfilter( h, 1.0, y_env)

        #slicer
        #calculate the mean of the signal
        mean = np.mean(y_filtered)
        #if the mean of the bit period is higher than the mean, the data is a 0
        rx_data = []
        sampled_signal = y_filtered[int(Fs/Fbit/2):len(y_filtered):int(Fs/Fbit)]
        for bit in sampled_signal:
            if bit > mean:
                rx_data.append(0)
            else:
                rx_data.append(1)
        return rx_data

    @staticmethod
    def diff_and_envelop_detector(x, Fs, Fbit, Fc, Fdev):
        # deffierentiator
        y_diff = np.diff(x,1)

        # create an envelope detector and then low-pass filter
        y_env = np.abs(sigtool.hilbert(y_diff))
        h=signal.firwin( numtaps=100, cutoff=Fbit*2, nyq=Fs/2)
        y_filtered=signal.lfilter( h, 1.0, y_env)

        return y_filtered


    @staticmethod
    def plot_signal(y,  Fs, Fbit):
        N_prntbits = 10  # number of bits to print in plots
        t = np.arange(0, float(N_prntbits) / float(Fbit), 1 / float(Fs), dtype=np.float)

        pl.plot(t[0:int(Fs * N_prntbits / Fbit)], y[0:int(Fs * N_prntbits / Fbit)])
        pl.xlabel('Time (s)')
        pl.ylabel('Amplitude (V)')
        pl.title('Amplitude vs Time')
        pl.show()
        return

    @staticmethod
    def bit_error(tx, rx):
        #print bit error
        N = len(tx)
        bit_error=0
        for i in range(0,len(tx)):
            if(rx[i] != tx[i]):
                bit_error = bit_error +  1

        return bit_error

    @staticmethod
    def mse(tx, rx):
        #print bit error
        N = len(tx)
        mse = np.sum(np.power(tx - rx, 2)) / N
        return mse


if __name__ == '__main__':
    #the following variables setup the system
    Fc = 1000       #simulate a carrier frequency of 1kHz
    Fbit = 50       #simulated bitrate of data
    Fdev = 500      #frequency deviation, make higher than bitrate
    N = 300          #how many bits to send
    Fs = 5000      #sampling frequency for the simulator, must be higher than twice the carrier frequency
    A_n = 0.1      #noise peak amplitude

    #generate some random data for testing
    x = np.array([random.randint (0,1) for n in range(N)])

    y = Communication.mod_fsk(x,Fs=Fs, Fbit=Fbit, Fc=Fc, Fdev=Fdev)
    #Communication.plot_signal(y, Fs, Fbit)

    y_n, snr = Communication.noise_channel(y, A_n)
    print("SNR = %fdB" % snr)
    #Communication.plot_signal(y_n, Fs, Fbit)

    rx = Communication.demod_fsk(y_n,Fs=Fs, Fbit=Fbit, Fc=Fc, Fdev=Fdev)
    print('error: ', Communication.bit_error(x, rx))
    print('mse:' , Communication.mse(y, y_n))




