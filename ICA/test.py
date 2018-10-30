#for x in range(len(self.times)):
#                print(self.times[x])

import numpy as np
import pylab as pl
import csv
from sklearn.decomposition import FastICA
"""
###############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 10, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal

S = np.c_[s1, s2]
#print("S:\n", S, "\n")
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data

A = np.array([[1, 1], [0.5, 2]])
#A = np.array([[0.125, 0.125], [0.0625, 0.25]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations\
# Compute ICA
ica = FastICA()
S_ = ica.fit(X).transform(X)  # Get the estimated sources
S_1 = S_[:, 0]
#print("S_: \n", S_, "\n")
#print("S_1: \n", S_1, "\n")
#S1_ = ica.fit(X1).transform(X1)
A_ = ica.mixing_  # Get estimated mixing matrix
print (A, "\n", "\n", A_)
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

###############################################################################
# Plot results
pl.figure()
pl.subplot(3, 1, 1)
pl.plot(S)
pl.title('True Sources')
pl.subplot(3, 1, 2)
pl.plot(X)
pl.title('X')
pl.subplot(3, 1, 3)
pl.plot(S_)
pl.title('ICA estimated sources')
pl.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
pl.show()

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 35
    lowcut = 0.5
    highcut = 3.0
    x = []

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 60
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.05
    f0 = 2.5
    x = 0.01 * np.sin(2 * np.pi * 1.0 * t)
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.sin(2 * np.pi * f0 * t)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()
"""
n = 10
start = n - 9
stop = range(10)
i = 0
for start in stop:
    i = i + 1
    print(i)
