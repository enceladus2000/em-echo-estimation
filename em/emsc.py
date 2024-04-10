import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy
from dataclasses import dataclass

from em.signal_utils import gen_ir
from typing import Tuple

@dataclass
class EMSC_Setup:
    # fs: int       # we will only use fs = 1 i.e. natural units
    num_ref: int    # number of reflections
    toa_range: Tuple[int, int]

    betas: np.ndarray = None

    max_iter: int = 100
    return_history: bool = False

    def __post_init__(self):
        if self.betas is None:
            self.betas = np.ones(self.num_ref) / self.num_ref
        
@dataclass
class EMSC_Estimate:
    gains: np.ndarray
    toas: np.ndarray    

    def sort_by_toas(self):
        sort_idx = np.argsort(self.toas)
        self.gains = self.gains[sort_idx]
        self.toas = self.toas[sort_idx]

def EMSC(src_signal, mic_signal, setup: EMSC_Setup, init_est: EMSC_Estimate):
    assert src_signal.shape[0] == mic_signal.shape[0], 'Source and microphone signals must have same length'
    est_iter = copy.deepcopy(init_est)
    if setup.return_history:
        est_hist = [est_iter]

    for i in range(setup.max_iter):
        est_iter = EMSC_iter(src_signal, mic_signal, setup, est_iter)
        
        # sort estimates by toa in est_iter
        est_iter.sort_by_toas()
        

        # TODO: EMSC_EstimateHist class for storing last N estimates as a queue or something
        if setup.return_history:
            est_hist.append(est_iter)

    if setup.return_history:
        return est_iter, est_hist
    else:
        return est_iter



def EMSC_iter(src_signal, mic_signal, setup: EMSC_Setup, cur_est: EMSC_Estimate):
    ssum = gen_ir(len(mic_signal), cur_est.toas, cur_est.gains)
    ssum = np.convolve(src_signal, ssum, mode='full')[:len(mic_signal)]

    new_est = EMSC_Estimate(np.zeros(setup.num_ref), np.zeros(setup.num_ref))
    for r in range(setup.num_ref):
        sr = gen_ir(len(mic_signal), [cur_est.toas[r]], [cur_est.gains[r]])
        sr = np.convolve(src_signal, sr, mode='full')[:len(mic_signal)]
        xmr = sr + setup.betas[r] * (mic_signal - ssum)

        new_est.gains[r], new_est.toas[r] = Mstep(xmr, src_signal, setup)

    return new_est

def Mstep(xmr, src_signal, setup: EMSC_Setup):
    """Returns gain and TOA estimate from M step for single reflection. """
    # this version is a faster version
    nmin, nmax = np.array(setup.toa_range, dtype=int) + len(src_signal) - 1
    Rxsfull = np.correlate(xmr, src_signal, mode='full')
    Rxs = Rxsfull[nmin:nmax]
    am = np.argmax(Rxs)
    maxtoa = am + nmin - len(src_signal) + 1
    maxgain = Rxs[am] / np.dot(src_signal, src_signal)
    # this version is dumb
    # Rxs = np.correlate(src_signal, xmr, mode='full')
    # am = np.argmax(Rxs)
    # maxtoa = am - len(src_signal) + 1
    # maxgain = Rxs[am] / np.dot(src_signal, src_signal)
    # (check math)
    # then do gradient descent on continuous function obtained from chosen fdf
    # (work out explicit GD steps) 
    return maxgain, maxtoa