import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

from em.sim_utils import gen_ir, delay_signal
from typing import Tuple

@dataclass
class EMSC_Setup:
    # fs: int       # we will only use fs = 1 i.e. natural units
    num_ref: int    # number of reflections
    toa_range: Tuple[int, int]

    betas: np.ndarray = None

    max_iter: int = 100
    return_history: bool = False
    Mstep_maximizer: str = 'bounded-brent'
    Mstep_maxiter: int = 30
    Mstep_xatol: float = 1e-3
    fdflen: int = 41    # length of fractional delay filter

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

SINCHALF = np.sinc(0.5)

def Mstep(xmr, src_signal, setup: EMSC_Setup):
    """Returns gain and TOA estimate from M step for single reflection. """
    nmin, nmax = np.array(setup.toa_range, dtype=int) + len(src_signal) - 1
    Rxsfull = np.correlate(xmr, src_signal, mode='full')
    Rxs = Rxsfull[nmin:nmax]
    src_signal_energy = np.dot(src_signal, src_signal)

    if setup.Mstep_maximizer.lower() == 'argmax':
        am = np.argmax(Rxs)
        maxtoa = am + setup.toa_range[0]
        maxgain = Rxs[am] / src_signal_energy

    elif setup.Mstep_maximizer.lower() == 'bounded-brent':
        global SINCHALF
        def J(x):
            # remember, x = actual TOA - TOAmin
            # TODO: smarter way to do using just Rxs?
            return -np.dot(xmr, delay_signal(src_signal, x+setup.toa_range[0], setup.fdflen))

        # find candidates for minimize_scalar from Rxs. 
        si = np.argsort(Rxs)[::-1]
        Rxsmax = Rxs[si[0]]
        # note: max(interpolated_s[i-0.5:i+0.5]) <= s[i] / sinc(0.5)
        cands = np.array(si[Rxs[si] > SINCHALF*Rxsmax], dtype=int)
        
        # remove adjacent candidates, replace with middle or earlier one
        candssorted = np.sort(cands)
        cands = {si[0]} # Rxsmax index is always a candidate
        i = 0
        while i < len(candssorted):
            if i+1 < len(candssorted) and candssorted[i+1] == candssorted[i] + 1:
                if i+2 < len(candssorted) and candssorted[i+2] == candssorted[i] + 2:
                    x0 = candssorted[i+1]
                    i += 3
                else:
                    x0 = candssorted[i]
                    i += 2
            else:
                x0 = candssorted[i]
                i += 1
            cands.add(x0)

        # remember, cands are indices in Rxs, thus actual TOA = c + TOAmin
        # Also, the optim result is in terms of index in Rxs, not actual TOA, since J's argument is assumed to be TOA - TOAmin

        Jmin = -Rxsmax
        maxtoa = si[0] + setup.toa_range[0]
        for c in cands:
            res = minimize_scalar(
                J, 
                method='Bounded', 
                bracket=(c-1, c+1), 
                bounds=(c-2, c+2),
                options={
                    'maxiter': setup.Mstep_maxiter, 
                    'disp': False,
                    'xatol': setup.Mstep_xatol,
                }
            )
            if res.success and res.fun < Jmin:
                Jmin = res.fun
                maxtoa = res.x + setup.toa_range[0]
        
        maxgain = -Jmin / src_signal_energy

    else:
        raise ValueError(f'Invalid Mstep_maximizer: {setup.Mstep_maximizer}')

    return maxgain, maxtoa