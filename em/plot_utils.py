import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy

def plot_irs_mc(irs: np.ndarray, plottype='single', window=None, **kwargs):
    if window is None:
        window = slice(None)
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        window = slice(*window)
    elif window.lower() == 'auto':
        raise NotImplementedError('Auto windowing not implemented yet')
    else:
        raise ValueError('Invalid window type')
    

    xaxis = np.arange(window.start, window.stop)
    if plottype.lower() == 'single':
        plt.plot(xaxis, irs[:, window].T, alpha=kwargs.get('alpha', 0.7), label=[f'mic {m}' for m in range(irs.shape[0])])
        plt.legend()
        plt.grid(True)
        plt.show()
    elif plottype.lower() in ('multi', 'subplots', 'subplot'):
        fig, axs = plt.subplots(irs.shape[0], 1, sharex=True, sharey=True, figsize=kwargs.get('multifigsize', (4, 6)))
        for m, ax in enumerate(axs):
            ax.plot(xaxis, irs[m, window])
            ax.set_title(f'Mic {m}')
        plt.show()
    else:
        raise ValueError('Invalid plot type')
    
