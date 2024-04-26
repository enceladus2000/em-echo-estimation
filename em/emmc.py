import numpy as np
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from scipy.signal import correlate as sp_correlate

from em.signal_utils import delay_signal
from typing import Tuple, Union

@dataclass
class EMMC_Setup:
	num_ref: int    # number of reflections
	num_mics: int   # number of microphones
	mic_array_radius: float	# in 'sampling wavelengths', i.e. wavelength of sound with frequency = fs
							# i.e. in natural units, 343 / fs meters
	   						# i.e. mic_array_radius = mic_array_radius_metres * fs / 343
	toa_range: Tuple[int, int]
	phi_range: Union[Tuple[int, int], Tuple[int, int, int]]	= None # (start, stop) or (start, stop, step), in radians

	betas: np.ndarray = None

	max_iter: int = 100
	return_history: bool = False
	Mstep_maximizer: str = 'gridsearch'
	# Mstep_maxiter: int = 30
	# Mstep_xatol: float = 1e-3
	fdflen: int = 41    		# length of fractional delay filter
	fdfwindow: str = 'blackman' # window for fractional delay filter

	def __post_init__(self):
		if self.betas is None:
			self.betas = np.ones(self.num_ref) / self.num_ref

		if self.phi_range is None:
			self.phi_range = (-np.pi, np.pi, 2*np.pi/180)
		elif len(self.phi_range) == 2:
			self.phi_range = (*self.phi_range, 2*np.pi/180)
		self.phi_grid = np.arange(*self.phi_range)

		self.toa_grid = np.arange(*self.toa_range)

# @dataclass 
class EMMC_Estimate:
	gains: np.ndarray	# (R,)
	toas: np.ndarray	# (R,)
	phis: np.ndarray	# (R,)
	tdoas: np.ndarray	# (M, R)

	def __init__(self, gains, toas, phis, setup: EMMC_Setup) -> None:
		self.gains = gains
		self.toas = toas
		self.phis = phis
		self.tdoas = compute_tdoas(phis, setup)

	def sort_by_toas(self):
		sort_idx = np.argsort(self.toas)
		self.gains = self.gains[sort_idx]
		self.toas = self.toas[sort_idx]
		self.phis = self.phis[sort_idx]
		if self.tdoas is not None:
			self.tdoas = self.tdoas[:, sort_idx]

	@classmethod
	def empty(cls, setup: EMMC_Setup):
		return cls(
			gains=np.zeros(setup.num_ref),
			toas=np.zeros(setup.num_ref),
			phis=np.zeros(setup.num_ref),
			setup=setup
		)
	
	def set_rth_reflection(self, r: int, toa: float, phi: float, gain: float, setup: EMMC_Setup):
		self.gains[r] = gain
		self.toas[r] = toa
		self.phis[r] = phi
		self.tdoas[:, r] = compute_tdoas(phi, setup)

	def __str__(self) -> str:
		return f'Gains: {self.gains}\nTOAs: {self.toas}\nDOAs: {self.phis}\nTDOAs:\n{self.tdoas}'
	def __repr__(self) -> str:
		return self.__str__()


def EMMC(src_signal, mic_signals, setup: EMMC_Setup, init_est: EMMC_Estimate):
	assert src_signal.shape[0] == mic_signals.shape[1], 'Source and microphone signals must have same length'

	est_iter = copy.deepcopy(init_est)
	if setup.return_history:
		est_hist = [est_iter]

	for i in range(setup.max_iter):
		est_iter = EMMC_iter(src_signal, mic_signals, setup, est_iter)
		est_iter.sort_by_toas()
		if setup.return_history:
			est_hist.append(est_iter)

	if setup.return_history:
		return est_iter, est_hist
	else:
		return est_iter

def EMMC_iter(src_signal, mic_signals, setup: EMMC_Setup, cur_est: EMMC_Estimate) -> EMMC_Estimate:
	xmrs = Estep(src_signal, mic_signals, setup, cur_est) 		# (M, R, N)

	new_est = EMMC_Estimate.empty(setup)
	for r in range(setup.num_ref):
		toa, phi, gain = Mstep_r(xmrs, src_signal, setup, r)
		new_est.set_rth_reflection(r, toa, phi, gain, setup)

	return new_est


def Mstep_r(xmr, src_signal, setup: EMMC_Setup, r: int):
	"""Returns TOA, DOA (phi) and gain (g) estimate from M step for single reflection (r). 

	Args:
		xmr (np.ndarray): (M, R, N) The **full** array of shifted signals
		src_signal (np.ndarray): Source signal (1D)
		setup (EMMC_Setup): Setup
		r (int): Index of reflection
	Returns:
		(float, float, float): TOA, DOA, gain
	"""
	if setup.Mstep_maximizer == 'gridsearch':
		cost_func = np.zeros((len(setup.toa_grid), len(setup.phi_grid)))
		for pidx, phicand in enumerate(setup.phi_grid):
			tdoacand = compute_tdoas(phicand, setup)

			Dxmsum = np.zeros(len(src_signal))
			for m in range(setup.num_mics):
				Dxmsum += delay_signal(xmr[m, r], -tdoacand[m], setup.fdflen, setup.fdfwindow)

			# for tidx, toacand in enumerate(setup.toa_grid):
			# 	shiftedsrcsig = delay_signal(src_signal, toacand, setup.fdflen, setup.fdfwindow)
			# 	cost_func[tidx, pidx] = np.dot(Dxmsum, shiftedsrcsig)
			nmin = setup.toa_range[0] + len(src_signal) - 1
			nmax = setup.toa_range[1] + len(src_signal) - 1
			cost_func[:, pidx] = np.correlate(Dxmsum, src_signal, mode='full')[nmin:nmax]
			# cost_func[:, pidx] = sp_correlate(Dxmsum, src_signal, mode='full')[nmin:nmax]

		toa_idx, phi_idx = np.unravel_index(np.argmax(cost_func), cost_func.shape)
		newtoa = setup.toa_grid[toa_idx]
		newphi = setup.phi_grid[phi_idx]
		newgain = cost_func[toa_idx, phi_idx] / np.dot(src_signal, src_signal) / setup.num_mics
	else:
		raise NotImplementedError(f'Mstep_maximizer={setup.Mstep_maximizer} not implemented')
	
	return newtoa, newphi, newgain

def Estep(src_signal, mic_signals, setup: EMMC_Setup, cur_est: EMMC_Estimate):
	xmrs = np.empty((setup.num_mics, setup.num_ref, len(src_signal))) # (M, R, N)
	# TODO: use gen_ir_uca instead?
	for m in range(setup.num_mics):
		# calc gain shifted signal for mic m and sum it
		gssm = np.array([
				cur_est.gains[r]*delay_signal(src_signal, cur_est.toas[r]+cur_est.tdoas[m, r], setup.fdflen, setup.fdfwindow) \
					for r in range(setup.num_ref)
			])
		gssm_sum = np.sum(gssm, axis=0)	# (R, N) -> sum -> (N,)

		for r in range(setup.num_ref):
			expected = gssm[r]
			delta = setup.betas[r]*(mic_signals[m] - gssm_sum)
			xmrs[m, r] = expected + delta

	return xmrs


def compute_tdoas(phi, setup: EMMC_Setup):
	"""Compute the (float) TDOAs for a given phi.

	Args:
		phi (float, or 1D array of floats): Angle(s) of arrival
		setup (Setup): Setup
		
	Returns:
		(setup.num_mics, ) np.ndarray: TDOAs
	OR	(setup.num_mics, setup.num_ref) np.ndarray: TDOAs (if phi is an array)
	"""
	if isinstance(phi, np.ndarray):
		res = [compute_tdoas(p, setup) for p in phi]
		return np.array(res).T
	elif isinstance(phi, (int, float)):
		tdoas = np.zeros((setup.num_mics, ))
		# TODO: memoize
		mic_angles = np.linspace(0, 2*np.pi, setup.num_mics, endpoint=False)

		for i in range(setup.num_mics):
			tdoas[i] = setup.mic_array_radius * (np.cos(mic_angles[0]-phi) - \
						np.cos(mic_angles[i]-phi))

		return tdoas
	else:
		raise TypeError(f'phi should be a float or a 1D array of floats, not {type(phi)}')
