import numpy as np
import matplotlib.pyplot as plt

# memoize!
def gen_frac_delay_kernel(delay, flen, window):
	"""Generates filter kernel for fractional delay using windowed sinc. Note:
	1. flen should be odd integer
	2. delay should be in [-0.5, 0.5]
	3. Kernel actually delays by delay + (flen - 1)/2, so account for that!

	Args:
		delay (float): Should be in [-0.5, 0.5], for filter symmetry.
		flen (int): Length of filter, recommended odd integer for symmetry. Higher the better.
		window (str): Type of window to apply to sinc function. Choose among 'hann', 'hamming', 'blackman', 'bartlett', or None (for no windowing).

	Returns
	"""
	assert flen % 2 == 1, 'Filter length should be odd integer'
	assert -0.5 <= delay <= 0.5, 'Delay should be in [-0.5, 0.5]'
	n = np.arange(flen)
	h = np.sinc(n - flen//2 - delay)
	# TODO: 0.5 delay is not symmetric, why?
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
intcount = 0
fraccount = 0
def delay_signal(signal, delay, flen, window='blackman'):
	"""Delays signal by real-valued delay using fractional delay filter. Note that the signal is delay by the actual delay, not delay + flen//2. Thus, delay must be greater than flen//2.

	Args:
		signal (np.ndarray): Signal to be delayed.
		delay (float): Real-valued delay. Must be greater than flen//2.
		flen (int): Length of fractional delay filter. Should be odd integer.
		window (str, optional): Kind of window. Defaults to 'blackman'.

	Returns:
		np.ndarray: Delayed signal of same length with actual delay.
	"""
	# assert delay >= flen//2, 'Delay should be greater than half filter length'
	# global intcount, fraccount
	intdly = int(np.round(delay))
	fracdly = delay - intdly
	if intdly >= 0:
		dlysignal = np.hstack((np.zeros(intdly), signal[:len(signal)-intdly]))
	elif intdly < 0:
		dlysignal = np.hstack((signal[-intdly:], np.zeros(-intdly)))
	if abs(fracdly) < 1e-5:
		# intcount += 1
		return dlysignal
	else:
		# fraccount += 1
		kernel = gen_frac_delay_kernel(fracdly, flen, window)
		return np.convolve(dlysignal, kernel, mode='same')

def add_white_noise(original_signal, snr_db) -> np.ndarray:
	"""Adds white noise to the original signal to achieve the desired SNR."""
	snr_linear = 10**(snr_db/10)
	p_noise = np.var(original_signal) / snr_linear
	noise = np.sqrt(p_noise) * np.random.randn(original_signal.shape[0])
	return original_signal + noise
