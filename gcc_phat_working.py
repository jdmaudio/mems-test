import sounddevice as sd
import numpy as np
import queue
import math
import threading
 # Based on https://github.com/xiongyihui/tdoa

class Microphone:
    def __init__(self, rate=16000, blocksize=1024, device=3, channels=2):
        self.queue = queue.Queue()
        self.quit_event = threading.Event()
        self.sample_rate = rate
        self.blocksize = blocksize 
        self.device = device
        self.channels = channels

    def _callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())
        # print(f"Buffer size: {len(indata)}")

    def read_chunks(self, size):
        with sd.InputStream(callback=self._callback, samplerate=self.sample_rate, blocksize=size, device=self.device, channels=self.channels, dtype='int16'):
            while not self.quit_event.is_set():
                frames = self.queue.get()
                if not frames.any():
                    break
                yield frames

    def close(self):
        self.quit_event.set()
        self.queue.put(np.array([]))


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc


def main():

    print(sd.query_devices())

    sample_rate = 48000
    device = 32
    channels = 2
    N = 4096 * 4

    mic = Microphone(sample_rate, N, device, channels)

    window = np.hanning(N)

    sound_speed = 343
    distance = 2.144 / 100

    max_tau = distance / sound_speed

    for data in mic.read_chunks(N):
        buf = np.fromstring(data, dtype='int16')
        tau, _ = gcc_phat(buf[0::channels]*window, buf[1::channels]*window, fs=sample_rate, max_tau=max_tau)
        theta = math.asin(tau / max_tau) * 180 / math.pi
        print('\ntheta: {}'.format(int(theta)))

        
if __name__ == '__main__':
    main()