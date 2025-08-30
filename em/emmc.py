import numpy as np
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from scipy.signal import correlate as sp_correlate
from scipy.optimize import minimize, OptimizeResult, Bounds
from em.signal_utils import delay_signal
from typing import Tuple, Union, Generator
from collections import namedtuple

@dataclass
class EMMC_Setup:
    num_ref: int  # number of reflections
    num_mics: int  # number of microphones
    mic_array_radius: float 
    # radius is in 'sampling wavelengths', i.e. wavelength of sound with frequency = fs
    # i.e. in natural units, 343 / fs meters
    # i.e. mic_array_radius = mic_array_radius_metres * fs / 343

    toa_range: Tuple[int, int]
    phi_range: Tuple[int, int, int] = None # (start, stop) or (start, stop, step), in radians

    betas: np.ndarray = None
    toa_atol: float = 1e-3
    phi_atol: float = np.deg2rad(0.05)

    max_iter: int = 100
    return_history: bool = False

    Mstep_maximizer: str = "gridsearch"
    Mstep_maxiter: int = 40

    fdflen: int = 41  # length of fractional delay filter
    fdfwindow: str = "blackman"  # window for fractional delay filter

    def __post_init__(self):
        if self.betas is None:
            self.betas = np.ones(self.num_ref) / self.num_ref

        if self.phi_range is None:
            self.phi_range = (-np.pi, np.pi, 2 * np.pi / 180)
        elif len(self.phi_range) == 2:
            self.phi_range = (*self.phi_range, 2 * np.pi / 180)
        self.phi_grid = np.arange(*self.phi_range)

        self.toa_grid = np.arange(*self.toa_range)


class EMMC_Estimate:
    gains: np.ndarray  # (R,)
    toas: np.ndarray  # (R,)
    phis: np.ndarray  # (R,)
    tdoas: np.ndarray  # (M, R)

    def __init__(self, gains, toas, phis, setup: EMMC_Setup) -> None:
        self.gains = np.array(gains)
        self.toas = np.array(toas)
        self.phis = np.array(phis)
        assert len(self.gains) == len(self.toas) == len(self.phis)
        self.tdoas = compute_tdoas(self.phis, setup)

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
            setup=setup,
        )

    def set_rth_reflection(
        self, r: int, toa: float, phi: float, gain: float, setup: EMMC_Setup
    ):
        self.gains[r] = gain
        self.toas[r] = toa
        self.phis[r] = phi
        self.tdoas[:, r] = compute_tdoas(phi, setup)

    def __str__(self) -> str:
        return f"Gains: {self.gains}\nTOAs: {self.toas}\nDOAs: {self.phis}\nTDOAs:\n{self.tdoas}"

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def num_ref(self):
        return len(self.gains)


# TODO: items in return_history don't necessarily need to be EMMC_estimate, best to have separate class... how to have class with optional/variable/on-the-fly attributes?
# TODO: type hints on shape and dtype of numpy arrays?
"""TODO: functional-program the shit out of this
- EMMC_iter as an actual generator
i.e. it returns a generator object
EMMCgen = EMMC_iter(args)
from funcy import nth
final_est = nth(setup.num_iter, EMMCgen)

however, there needs to be an encapsulating function taking care of convergence of ests, estimate history, dupvals, etc... or a class?
- EMMC_iter can be two fucking lines lmao (but idk, need to check about parallel computation-ing it)
- explicit cost function method and grid in Mstep_r

grid = J(toa, doa) for toa in toa_grid for doa in doa_grid # numpy way of doing this?
grid -> debug!
est = gridsearch(grid)
OR
est = optimizer(J(toa, doa) for ...)

"""

EchoEstimate = namedtuple('Echo', ['gain', 'toa', 'phi'])
@dataclass
class EchoEstimate:
    gain: float
    toa: float
    phi: float

    def equal(self, other: EchoEstimate, setup: EMMC_Setup):
        return np.isclose(self.toa, other.toa, atol=setup.toa_atol) \
            and np.isclose(self.phi, other.phi, atol=setup.phi_atol)
    
    def combine(self, other: EchoEstimate):
        self.gain = self.gain + other.gain
        self.toa = (self.toa + other.toa) / 2
        self.phi = (self.phi + other.phi) /2

    @classmethod
    def random(cls, setup: EMMC_Setup) -> EchoEstimate:
        cls.gain = np.random.uniform(0, 1.)
        cls.toa = np.random.uniform(*setup.toa_range)
        cls.phi = np.random.uniform(setup.phi_range[0], setup.phi_range[1])
        return cls

# TODO: I should have EchoEstimate class, store each estimate as list of estimates instead?
def replace_dups(est: EMMC_Estimate, setup: EMMC_Setup) -> EMMC_Estimate:
    echoes = (EchoEstimate(est.gains[i], est.toas[i], est.phis[i]) for i in range(est.num_ref))
    new_echoes = [next(echoes)]
    for e in echoes:
        if e.equal(new_echoes[-1], setup):
            new_echoes[-1].combine(e)
        else:
            new_echoes.append(e)

    new_echoes += [EchoEstimate.random(setup) for i in range(est.num_ref - len(new_echoes))]
    new_est = EMMC_Estimate(
        gains=np.array([e.gain for e in new_echoes]),
        toas=np.array([e.toa for e in new_echoes]),
        phis=np.array([e.phi for e in new_echoes]),
        setup=setup
    )
    new_est.
    return new_est


def EMMC(
    src_signal, mic_signals, setup: EMMC_Setup, init_est: EMMC_Estimate
) -> Generator[EMMC_Estimate, None, None]:
    # exposes emmc, does other stuff like
    # checking for convergence
    # checking and addressing dupvals
    emmc_gen = EMMC_simple(src_signal, mic_signals, setup, init_est)
    for est in emmc_gen:
        est = replace_dups(est)
        yield est

# TODO: rename to _EMMC cuz this is vanilla - no convergence checking, dupval, etc. Another encapsulant will do this
def EMMC_simple(
    src_signal, mic_signals, setup: EMMC_Setup, init_est: EMMC_Estimate
) -> Generator[EMMC_Estimate, None, None]:
    assert (
        src_signal.shape[0] == mic_signals.shape[1]
    ), "Source and mic signals must have same length"

    q = EstimateQueue(setup) # setup.atol's, setup.convergence_checker_window
    est_iter = copy.deepcopy(init_est)
    for i in range(setup.max_iter):  # TODO: maybe infinite loop?
        est_iter = EMMC_iter(src_signal, mic_signals, setup, est_iter)
        est_iter = replace_dups(est_iter, setup)
        q.add(est_iter)
        if check_convergence(q):
            break
        yield est_iter
        # why are we sorting estimates anyway? 1. for dupval 2. when plotting/analysing history/checking for convergence of estimates, we need some 'continuity' in the estimates, i.e. the est-true reflection association should be consistent.
        # if we aren't sorting, the non-duplicate values MUST remain in their same spot -> the dupval replacements must go in the same spot as the replaced values.


def EMMC_iter(src_signal, mic_signals, setup: EMMC_Setup, cur_est: EMMC_Estimate):
    xmrs = Estep(src_signal, mic_signals, setup, cur_est)  # (M, R, N)

    new_est = EMMC_Estimate.empty(setup)
    for r in range(setup.num_ref):
        toa, phi, gain = Mstep_r(xmrs[:, r], src_signal, setup)
        new_est.set_rth_reflection(r, toa, phi, gain, setup)

    new_est.sort_by_toas()
    return new_est


def Mstep_r(xmr, src_signal, setup: EMMC_Setup):
    """Returns TOA, DOA (phi) and gain (g) estimate from M step for single reflection (r).

    Args:
                    xmr (np.ndarray): (M, N) The array of shifted signals, all mics for a single reflection
                    src_signal (np.ndarray): Source signal (1D)
                    setup (EMMC_Setup): Setup
                    r (int): Index of reflection
    Returns:
                    (float, float, float): TOA, DOA, gain
    """
    # M_func_p = lambda toa, doa: M_func(toa, doa, src_signal, xmr, setup)
    # TODO: rename gridsearch to something else?
    if setup.Mstep_maximizer == "gridsearch":
        return M_gridsearch(xmr, src_signal, setup)

    elif setup.Mstep_maximizer == "nelder-mead":
        # obtain rough gridsearch toa and phi estimate
        # not worried about anything but the first (toa, phi) candidate, maybe can do later.
        grid_toa, grid_phi = M_gridsearch(xmr, src_signal, setup, calc_gain=False)

        costfunc = lambda x: -M_func(*x, src_signal, xmr, setup)

        delta_phi = setup.phi_range[2]
        res: OptimizeResult = minimize(
            costfunc,
            (grid_toa, grid_phi),
            method="nelder-mead",
            bounds=Bounds(
                lb=(grid_toa-1, grid_phi - delta_phi),
                ub=(grid_toa+1, grid_phi + delta_phi)
            ),
            options={
                "maxiter": setup.Mstep_maxiter,
                "xatol": setup.toa_atol,
            },
        )
        if res.success:
            M_min = res.fun
            maxtoa, maxphi = res.x
            # print(f"Took {res.nit} iterations to minimize")
        else:
            print(f"Minimization failed, msg: {res.message}")

            M_min = costfunc((grid_toa, grid_phi))
            maxtoa, maxphi = grid_toa, grid_phi
        maxgain = -M_min / np.dot(src_signal, src_signal) / setup.num_mics
        return maxtoa, maxphi, maxgain

    else:
        raise NotImplementedError(
            f"Mstep_maximizer={setup.Mstep_maximizer} not implemented"
        )


# xmr = xmrs[:, r], shape = (M, N)
def M_func(toa, doa, src_signal, xmr, setup: EMMC_Setup) -> float:
    tdoas = compute_tdoas(doa, setup)
    Dxmsum = np.zeros(len(src_signal))
    for m in range(setup.num_mics):
        Dxmsum += delay_signal(xmr[m], -tdoas[m], setup.fdflen, setup.fdfwindow)
    return np.dot(Dxmsum, delay_signal(src_signal, toa, setup.fdflen, setup.fdfwindow))


def M_gridsearch(xmr, src_signal, setup: EMMC_Setup, calc_gain=True):
    cost_func = np.zeros((len(setup.toa_grid), len(setup.phi_grid)))
    nmin = setup.toa_range[0] + len(src_signal) - 1
    nmax = setup.toa_range[1] + len(src_signal) - 1
    for pidx, phicand in enumerate(setup.phi_grid):
        tdoacand = compute_tdoas(phicand, setup)

        Dxmsum = np.zeros(len(src_signal))
        for m in range(setup.num_mics):
            Dxmsum += delay_signal(xmr[m], -tdoacand[m], setup.fdflen, setup.fdfwindow)

        cost_func[:, pidx] = np.correlate(Dxmsum, src_signal, mode="full")[nmin:nmax]

    toa_idx, phi_idx = np.unravel_index(np.argmax(cost_func), cost_func.shape)
    newtoa = setup.toa_grid[toa_idx]
    newphi = setup.phi_grid[phi_idx]
    # TODO: memoize src_signal^2 somehow
    if calc_gain:
        newgain = (
            cost_func[toa_idx, phi_idx]
            / np.dot(src_signal, src_signal)
            / setup.num_mics
        )
        return newtoa, newphi, newgain
    else:
        return newtoa, newphi


def Estep(src_signal, mic_signals, setup: EMMC_Setup, cur_est: EMMC_Estimate):
    xmrs = np.empty((setup.num_mics, setup.num_ref, len(src_signal)))  # (M, R, N)
    # TODO: use gen_ir_uca instead?
    for m in range(setup.num_mics):
        # calc gain shifted signal for mic m and sum it
        gssm = np.array(
            [
                cur_est.gains[r]
                * delay_signal(
                    src_signal,
                    cur_est.toas[r] + cur_est.tdoas[m, r],
                    setup.fdflen,
                    setup.fdfwindow,
                )
                for r in range(setup.num_ref)
            ]
        )
        gssm_sum = np.sum(gssm, axis=0)  # (R, N) -> sum -> (N,)

        for r in range(setup.num_ref):
            expected = gssm[r]
            delta = setup.betas[r] * (mic_signals[m] - gssm_sum)
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
        tdoas = np.zeros((setup.num_mics,))
        # TODO: memoize
        mic_angles = np.linspace(0, 2 * np.pi, setup.num_mics, endpoint=False)

        for i in range(setup.num_mics):
            tdoas[i] = setup.mic_array_radius * (
                np.cos(mic_angles[0] - phi) - np.cos(mic_angles[i] - phi)
            )

        return tdoas
    else:
        raise TypeError(
            f"phi should be a float or a 1D array of floats, not {type(phi)}"
        )
