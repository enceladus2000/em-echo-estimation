import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy
from typing import List
from em.emmc import EMMC_Estimate

# def plot_history_mc(hist: List[EMMC_Estimate], gt: EMMC_Estimate):
#     fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
#     axs[0].plot([h.gains for h in hist], label=[f'mic {m}' ]


def plot_signals_mc(irs: np.ndarray, plottype='single', window=None, **kwargs):
    if window is None:
        window = slice(None)
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        window = slice(*window)
    elif window.lower() == 'auto':
        raise NotImplementedError('Auto windowing not implemented yet')
    else:
        raise ValueError('Invalid window type')
    
    xaxis = np.arange(
        window.start if window.start is not None else 0, 
        window.stop if window.stop is not None else irs.shape[1],
    )
    if plottype.lower() == 'single':
        plt.plot(xaxis, irs[:, window].T, alpha=kwargs.get('alpha', 0.7), label=[f'mic {m}' for m in range(irs.shape[0])])
        plt.legend()
        plt.grid(True)
    elif plottype.lower() in ('multi', 'subplots', 'subplot'):
        fig, axs = plt.subplots(irs.shape[0], 1, sharex=True, sharey=True, figsize=kwargs.get('multifigsize', (4, 6)))
        for m, ax in enumerate(axs):
            ax.plot(xaxis, irs[m, window])
            ax.set_title(f'Mic {m}')
    else:
        raise ValueError('Invalid plot type')
    
    plt.tight_layout()
    plt.show()
    
