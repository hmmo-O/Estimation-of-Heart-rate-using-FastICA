import numpy as np
import time
import cv2
import pylab
import os
import sys
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, firwin
import datetime
import csv

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):
    
    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 30
        self.buffer_size = 250
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []

        self.test1 = []
        self.test2 = []
        self.test3 = []
        self.times1 = []
        self.B = []
        self.G = []
        self.R = []
        self.S = []
        self.S_ = []
        self.count = 0
        self.flag = 0
        self.X = 0

        self.freqs = []
        self.fftA = []
        self.fftH = []

        self.slices = [[0]]
        self.t0 = time.time()
        self.bpmsA = []
        self.bpmsH = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.find_faces = True

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]
                
    def train(self):
        self.trained = not self.trained
        return self.trained
    """    
    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        xrange = range
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()
    """  
    def smooth(self, x,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
    
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
    
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

        output:
            the smoothed signal
        
        example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)
    
        see also: 
    
            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter
 
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError('Smooth only accepts 1 dimension arrays.')

        if x.size < window_len:
            raise ValueError('Input vector needs to be bigger than window size.')


        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def hangover_time_filter(self):
        #Hangover-time Filtering
        if len(self.bpmsH) < 10:
            #print(self.bpm, " ", L)
            self.fftH.append(self.bpm)
        
        if self.flag == 0:
            if len(self.bpmsH) >= 10:
                t0 = len(self.bpmsH)
                t0 = t0 - 1
                T = abs(self.bpmsH[t0 - 1] - self.bpmsH[t0])
                if T < 10:
                    #print("<10 bpm: ", self.bpmsH[t0], " ", L)
                    self.fftH.append(self.bpmsH[t0])
                elif T > 10:
                    self.X = self.bpmsH[t0]
                    self.flag = 1
        elif self.flag == 1:
            self.count = self.count + 1
            if self.count % 2 == 0: #for every 2 seconds
                new_X = self.bpmsH[len(self.bpmsH) - 1]
                compare = abs(new_X - self.X)
                if compare <= 10:
                    #print("new_X <10bpm: ", new_X, " ", L)
                    self.fftH.append(new_X)
                    self.flag = 0
                elif compare > 10:
                    #print("X >10bpm: ", self.X, " ", L)
                    self.bpmsH[-1] = self.X #replacing the last item with new item
                    self.fftH.append(self.X)
                    self.flag = 0
            else:
                self.fftH.append(self.fftH[-1])

    def Adaptive_Filter(self):
        #Adaptive Filtering
        if len(self.bpmsA) < 10:
            #print(self.bpm, " ", L)
            self.fftA.append(self.bpm)                     
        elif len(self.bpmsA) >= 10:
            t0 = len(self.bpmsA) - 1
            T = abs(self.bpmsA[t0 - 1] - self.bpmsA[t0])
            if T < 10:
                #print("<10 bpm: ", self.bpmsA[t0], " ", L)
                self.fftA.append(self.bpmsA[t0])
            elif T > 10:
                b = 0
                t = t0 - 10
                #t0 = range(t0)
                for i in range(t, t0):
                    b = b + self.bpmsA[i]
                new_bpm = b/10
                #print("b/10: ", new_bpm, " ", L)
                self.bpmsA[-1] = new_bpm
                self.fftA.append(new_bpm)

    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:
            """
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                    cam),
                (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                       (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                       (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
            """
            self.data_buffer, self.times, self.trained = [], [], False
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))
            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])

                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]

            forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)

            self.draw_rect(self.face_rect, col=(255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                       (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            cv2.putText(self.frame_out, "Forehead",
                       (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        """
        cv2.putText(
            self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                cam),
            (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
        cv2.putText(
            self.frame_out, "Press 'S' to restart",
                   (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                   (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        """
        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1)

        x, y, w, h = forehead1
        subframe = self.frame_in[y:y + h, x:x + w, :]
        B = np.mean(subframe[:, :, 0])
        G = np.mean(subframe[:, :, 1])
        R = np.mean(subframe[:, :, 2])
        
        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 30.
        lowcut = 0.5
        highcut = 3.0


        ica = FastICA(whiten = True)
        self.B.append(B)
        self.G.append(G)
        self.R.append(R)
        #L = 3*fs
        for L in range(1800):
            if len(self.B) == L*fs:
                if len(self.G) == L*fs:
                    if len(self.R) == L*fs:
                        
                        self.times.append(time.time() - self.t0)

                        """
                        plt.figure(1)
                        plt.clf()
                        
                        plt.subplot(3, 1, 1)
                        plt.plot(self.B, color = 'blue')
                        plt.title('Blue original')

                        plt.subplot(3, 1, 2)
                        plt.plot(self.G, color = 'green')
                        plt.title('Green original')

                        plt.subplot(3, 1, 3)
                        plt.plot(self.R, color = 'red')
                        plt.title('Red original')
                        """
                        #fir = firwin(1800, [0.5, 3.0], fs = fs, pass_zero=False)
                        #print(fir)           
                        #B_bp = np.dot(fir, self.B)
                        #G_bp = np.dot(fir, self.G)
                        #R_bp = np.dot(fir, self.R)"""
                    
                        B_bp = self.butter_bandpass_filter(self.B, lowcut, highcut, fs, order=6)
                        G_bp = self.butter_bandpass_filter(self.G, lowcut, highcut, fs, order=6)
                        R_bp = self.butter_bandpass_filter(self.R, lowcut, highcut, fs, order=6)

                        """
                        plt.figure(2)                    
                        plt.plot(B_bp, label='Filtered signal (blue)', color = 'blue')
                        plt.plot(G_bp, label='Filtered signal (green)', color = 'green')
                        plt.plot(R_bp, label='Filtered signal (red)', color = 'red')
                        plt.xlabel('samples')
                        plt.grid(True)
                        plt.axis('tight')
                        plt.legend(loc='upper left')
                        """

                        #A = np.array([[10.44, 19.99, 35.93], [-1.16, 16.68, 38.99], [1.69, 10.24, 41.14]])

                        self.S = np.c_[B_bp, G_bp, R_bp]
                        self.S /= self.S.std(axis=0) # Standardize data
                        X = self.S
                        #X = np.dot(self.S, A.T) 
                        self.S_ = ica.fit(X).transform(X)  # Get the estimated sources
                        A_ = ica.mixing_  # Get estimated mixing matrix       
                        #print(A_)
                        assert np.allclose(X, np.dot(self.S_, A_.T) + ica.mean_)

                        """
                        plt.figure(3)
                        models = [X, self.S_]
                        names = ['Observations (mixed signal)', 'ICA recovered signals']
                        colors = ['blue', 'green', 'red']
                        for ii, (model, name) in enumerate(zip(models, names), 1):
                            plt.subplot(2, 1, ii)
                            plt.title(name)
                            for sig, color in zip(model.T, colors):
                                plt.plot(sig, color=color)

                        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
                    
                        plt.figure(4)
                        plt.subplot(3, 1, 1)
                        plt.plot(X[:, 2], color = 'red')
                        plt.title('Filtered Red')
                        
                        plt.subplot(3, 1, 2)
                        plt.plot(X[:, 1], color = 'green')
                        plt.title('Filtered Green')
                        
                        plt.subplot(3, 1, 3)
                        plt.plot(X[:, 0], color = 'blue')
                        plt.title('Filtered Blue')
                        
                        plt.figure(5)
                        plt.subplot(3, 1, 1)
                        plt.plot(self.S_[:, 2], color = 'red')
                        plt.title('Red ICA')
                        
                        plt.subplot(3, 1, 2)
                        plt.plot(self.S_[:, 1], color = 'green')
                        plt.title('Green ICA')
                        
                        plt.subplot(3, 1, 3)
                        plt.plot(self.S_[:, 0], color = 'blue')
                        plt.title('Blue ICA')
                        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
                        """

                        y = self.smooth(self.S_[:, 0], window_len = 11, window = 'flat')
                        y1 = self.smooth(self.S_[:, 1], window_len = 11, window = 'flat')
                        y2 = self.smooth(self.S_[:, 2], window_len = 11, window = 'flat')

                        raw = np.fft.rfft(y) #blue
                        raw1 = np.fft.rfft(y1) #green
                        raw2 = np.fft.rfft(y2) #red

                        fft = np.abs(raw) #blue
                        fft1 = np.abs(raw1) #green
                        fft2 = np.abs(raw2) #red

                        """
                        plt.figure(6)
                        plt.subplot(3, 1, 1)
                        plt.plot(fft2, color = 'red')
                        plt.title('Red FFT')
                        
                        plt.subplot(3, 1, 2)
                        plt.plot(fft1, color = 'green')
                        plt.title('Green FFT')
                        
                        plt.subplot(3, 1, 3)
                        plt.plot(fft, color = 'blue')
                        plt.title('Blue FFT')
                        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
                        """

                        #freqfft = [np.argmax(fft), np.argmax(fft1), np.argmax(fft2)]
                        #print(np.argmax(fft2), " Red\n", np.argmax(fft1), " Green\n", np.argmax(fft), " Blue")

                        #plt.show()

                        freqfft = [np.argmax(fft), np.argmax(fft1), np.argmax(fft2)]
                            
                                
                        #print(np.argmax(fft2), "\n", np.argmax(fft1), "\n", np.argmax(fft))
                        #print(60/L * max(freqfft))

                        self.bpm = 60/L*max(freqfft)
                        #self.bpm = max(freqfft)
                        #self.fft = self.bpm
                        
                        #self.bpm = max(freqfft)
                        self.bpmsA.append(self.bpm)
                        self.bpmsH.append(self.bpm)

                        self.hangover_time_filter()
                        self.Adaptive_Filter()
                        
                        print(self.fftA[-1], "--> A", self.fftH[-1], "--> H", L)
                        self.freqs.append(L)
                        
                        """freqs = 60. * self.freqs

                        idx = np.where((freqs > 50) & (freqs < 200))

                        pfreq = freqs[idx]
                        self.freqs = pfreq"""
                
        
        #X = np.c_(self.B, self.G)
        
        #G_ = ica.fit(self.G).transform(self.G)  # Get the estimated sources
        #R_ = ica.fit(self.R).transform(self.R)  # Get the estimated sources
        

        #A_ = ica.mixing_
        #vals = self.get_subface_means(forehead1)
        
        
        """self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
         
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
               
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
       
        self.samples = processed
        
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            
            self.times1 = even_times
            self.test1 = interpolated

            interpolated = np.hamming(L) * interpolated
            
            self.test2 = interpolated

            interpolated = interpolated - np.mean(interpolated)

            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs

            idx = np.where((freqs > 50) & (freqs < 200))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            #print (self.bpm)
            self.idx += 1

            
            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
        
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps
            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(%0.1f bpm)" % (self.bpm)
            tsize = 1
            cv2.putText(self.frame_out, text,
                       (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)"""