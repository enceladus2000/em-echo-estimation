import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from em.emmc import compute_tdoas, EMMC_Setup
from em.signal_utils import gen_frac_delay_kernel

def gen_ir_uca(irlen, toas, doas, amps, setup: EMMC_Setup, fdflen=41, fdfwin='blackman'):
    """Generate multi-channel IR for UCA setup.

    Args:
        irlen (int): Length of each IR signal.
        toas (np.ndarray): (R,) array of TOAs for each reflection.
        doas (np.ndarray): (R,) array of DOAs for each reflection.
        amps (np.ndarray): (R,) array of amplitudes for each reflection.
        setup (EMMC_Setup): Needs num_mics and mic_array_radius.
        fdflen (int, optional): Length of fractional delay filter window. Defaults to 41.
        fdfwin (str, optional): Type of FDF window. Defaults to 'blackman'.

    Returns:
        np.ndarray: (M, irlen) array of IRs for each mic. 
    """
    tdoas = compute_tdoas(doas, setup)  # (M, R)
    irs = np.empty((setup.num_mics, irlen))
    for m in range(setup.num_mics):
        irs[m] = gen_ir(irlen, toas + tdoas[m], amps, fdflen, fdfwin)
    return irs

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