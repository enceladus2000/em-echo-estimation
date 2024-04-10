import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy
from scipy.signal import chirp

def gen_frac_delay_kernel(delay, flen, window):
    """Generates filter kernel for fractional delay using windowed sinc. Note:
    1. flen should be odd integer
    2. delay should be in [-0.5, 0.5]
    3. Kernel actually delays by delay + (flen - 1)/2, so account for that!

    Args:
        delay (float): Recommended in [-0.5, 0.5], for filter symmetry.
        flen (int): Length of filter, recommended odd integer for symmetry. Higher the better.
        window (str): Type of window to apply to sinc function. Choose among 'hann', 'hamming', 'blackman', 'bartlett', or None (for no windowing).

    Returns
    """
    n = np.arange(flen)
    h = np.sinc(n - flen//2 - delay)
    # TODO: shouldn't windows be applied to sinc function before delay? 
    if window == 'blackman':
        h *= np.blackman(flen)
    elif window in ('hann', 'hanning'):
        h *= np.hanning(flen)
    elif window == 'hamming':
        h *= np.hamming(flen)
    elif window == 'bartlett':
        h *= np.bartlett(flen)
    elif window not in (None, 'rectangular'):
        raise ValueError('Unknown window type')
    
    return h

def delay_signal(signal, delay, flen, window='blackman'):
    kernel = gen_frac_delay_kernel(delay, flen, window)
    return np.convolve(signal, kernel, mode='same') # TODO: shouldn't i be using full mode?

# REFACTOR: amps first, delays next
def gen_ir(irlen, delays, amps, fdflen=41, fdfwin='blackman'):
    delays = np.array(delays)
    intdlys = np.round(delays).astype(int)
    fracdlys = delays - intdlys

    ir = np.zeros(irlen)
    for id, fd, amp in zip(intdlys, fracdlys, amps):
        kernel = gen_frac_delay_kernel(fd, fdflen, fdfwin)
        i1 = id - fdflen//2
        i2 = i1 + fdflen
        if i1 < 0 or i2 > irlen:
            raise ValueError('frac delay filter too long for given delays ', intdlys)
        ir[i1:i2] += amp * kernel

    return ir

def gen_chirp(f0, f1, chirp_len, signal_len):
    """Generate a chirp signal with frequency sweep from f0 to f1.

    Args:
        f0 (int): Starting frequency in samples per second.
        f1 (int): End frequency in samples per second.
        length (int): Length of non-silent part of signal in samples.
        silence (int): Length of appended silence in samples.
    """
    assert chirp_len <= signal_len, 'Chirp length should be less than signal length'
    t = np.linspace(0, chirp_len, chirp_len, endpoint=False)
    sig = chirp(t, f0, chirp_len, f1, method='linear', phi=-90)
    return np.hstack((sig, np.zeros(signal_len-chirp_len)))

def add_white_noise(original_signal, snr_db) -> np.ndarray:
	"""Adds white noise to the original signal to achieve the desired SNR."""
	snr_linear = 10**(snr_db/10)
	p_noise = np.var(original_signal) / snr_linear
	noise = np.sqrt(p_noise) * np.random.randn(original_signal.shape[0])
	return original_signal + noise

if __name__ == '__main__':
    delays = [111.2, 145.1, 177.5]
    amps = [1, 0.5, 0.3]
    ir = gen_ir(300, delays, amps, fdflen=41, fdfwin='blackman')

    plt.plot(ir)
    plt.show()
