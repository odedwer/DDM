## ConstModel param recovery
# Fittables are - drift rate (constant), noise (constant), overlay (constant)
# Constants (not fittable) - boundaries, starting point (0)

import typing
from itertools import product, combinations

import numpy as np
import matplotlib.pyplot as plt
import ddm
from ddm import Model, Fittable, Solution
from ddm.functions import fit_adjust_model
from ddm.models import LossRobustBIC, DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision
from tqdm import tqdm
import pandas as pd
from ddm import set_N_cpus

# set_N_cpus(4)
I = 10

DEFAULT_N_SAMPLES = 1000


def sample_model(model, n_samples=DEFAULT_N_SAMPLES) -> ddm.Sample:
    """
    get samples from a model - samples are trials
    :param model: The model to generate trials by
    :param n_samples: optional, defaults to DEFAULT_N_SAMPLES.
                      The number of trials to generate
    :return: ddm.Sample
    """
    sol: Solution = model.solve()
    samples = sol.resample(n_samples)
    return samples


def fit_model(model, **kwargs):
    if "samples" not in kwargs.keys():
        kwargs["samples"] = sample_model(model) if "n_samples" not in kwargs.keys() else sample_model(model, kwargs[
            "n_samples"])
    fit_adjust_model(kwargs["samples"], model,
                     fitting_method="differential_evolution",
                     lossfunction=LossRobustBIC, verbose=False)


def param_recovery(fittable_model: Model, param_defaults_dict: dict, param_factory_dict: dict, param_ranges: dict) -> \
        typing.Tuple[np.array, np.array, list]:
    sorted_ranges = np.array(list(param_ranges.values()))[np.argsort(list(param_ranges.keys()))]
    param_combs = np.array(list(product(*sorted_ranges)))
    model_params = param_defaults_dict.copy()
    fit_param_names = list(param_ranges.keys())
    model_param_names = list(param_factory_dict.keys())
    fit_results = np.full_like(param_combs, fill_value=np.nan)
    for i, param_comb in enumerate(tqdm(param_combs)):
        for j, param_name in enumerate(fit_param_names):
            exec('model_params[model_param_names[j]] = param_factory_dict[model_param_names[j]](%s = %f)' % (
                param_name, param_comb[j]))
        _, fit_results[i, :] = single_run_param_recovery(fittable_model, model_params)
    # plotting simulated vs recovered parameters
    plot_real_vs_recovered_params(fit_param_names, fit_results, param_combs, param_ranges)

    # plotting recovery success of parameter interactions
    fig = plt.figure()
    idx = np.array(list(product(range(len(fit_param_names)))))
    df = pd.DataFrame(data=np.hstack([param_combs, fit_results]),
                      columns=fit_param_names.extend(["fitted_" + s for s in fit_param_names]))
    loss = lambda real, fitted: (real - fitted) ** 2
    # z score data
    means = df.iloc[:, 0:len(fit_param_names)].mean()
    stds = df.iloc[:, 0:len(fit_param_names)].std()
    df.iloc[:, 0:len(fit_param_names)] = (df.iloc[:, 0:len(fit_param_names)] - means) / stds
    df.iloc[:, len(fit_param_names):2 * len(fit_param_names)] = (df.iloc[:,
                                                                 0:len(fit_param_names)] - means) / stds
    loss_df = loss(df.iloc[:, 0:len(fit_param_names)], df.iloc[:, 0:len(fit_param_names)])
    df = pd.concat([df, loss_df], axis=1)

    idx_combs = list(combinations(range(len(fit_param_names)), 2))
    for idx in idx_combs:
        idx = np.array(idx)
        grouped_df: pd.DataFrame = df.groupby(by=idx).mean()
        mse = grouped_df.iloc[:, (2 * len(fit_param_names)) + idx].to_numpy()
        mse = np.sqrt(mse).mean(1)
        # TODO: unfold this to a 2D matrix for imshow

    return fit_results, param_combs, fit_param_names


def plot_real_vs_recovered_params(ctor_param_names, fit_results, param_combs, param_ranges) -> typing.Tuple[
    plt.Figure, plt.Axes]:
    """
    Plots the recovered parameters vs the real parameters used for simulation.
    :param ctor_param_names: Names of the recovered parameters
    :param fit_results: the recovered parameters
    :param param_combs: the real parameters used for simulation
    :param param_ranges: the range of parameters used for simulation
    :return: figure & axes
    """
    fig = plt.figure()
    for i, param_name in enumerate(ctor_param_names):
        ax = fig.add_subplot(1, len(ctor_param_names), i + 1)
        ax.scatter(param_combs[:, i], fit_results[:, i])
        ax.set_title(param_name)
        plot_lim = np.linspace(param_ranges[param_name].min(), param_ranges[param_name].max(), 2)
        ax.plot(plot_lim, plot_lim, '--')
    return fig, ax


def single_run_param_recovery(fittable_model: Model, model_params, n_samples=DEFAULT_N_SAMPLES) -> typing.Tuple[
    np.array, np.array]:
    """

    :param fit_model:
    :param kwargs: Should contain the following:
    model_params:dict with the following keys with the values matching the model:
                        name=,
                        drift=,
                        noise=,
                        bound=,
                        overlay=,
                        dx=DX,
                        dt=DT,
                        T_dur=T_DUR
    :return:
    """
    sim_model = Model(**model_params)
    samples = sample_model(sim_model, n_samples)
    fit_model(fittable_model, samples=samples)
    fitted_params = np.array(list(map(lambda a: float(a), fittable_model.get_model_parameters())))
    param_names = np.array(fittable_model.get_model_parameter_names())
    sorting_idxs = np.argsort(param_names)
    return param_names[sorting_idxs], fitted_params[sorting_idxs]


T_DUR = 2
DT = .01
DX = .001
DRIFT_CONST_FIT = {"minval": 0, "maxval": 4}
NOISE_CONST_FIT = {"minval": .5, "maxval": 4}
OVERLAY_CONST_FIT = {"minval": 0, "maxval": 1}
BOUND = 1.1

const_model_sim_defaults = {"dx": DX, "dt": DT, "T_dur": T_DUR, "bound": BoundConstant(B=BOUND)}
const_model_fittable = Model(name='const_model_fittable',
                             drift=DriftConstant(drift=Fittable(**DRIFT_CONST_FIT)),
                             noise=NoiseConstant(noise=Fittable(**NOISE_CONST_FIT)),
                             bound=BoundConstant(B=BOUND),
                             overlay=OverlayNonDecision(nondectime=Fittable(**OVERLAY_CONST_FIT)),
                             dx=DX, dt=DT, T_dur=T_DUR)

# %%
# const_model_params = {"drift": DriftConstant(drift=2), "noise": NoiseConstant(noise=1.5),
#                       "bound": BoundConstant(B=BOUND), "overlay": OverlayNonDecision(nondectime=.3)}

const_model_factory_dict = {"drift": DriftConstant, "noise": NoiseConstant, "overlay": OverlayNonDecision}

# params = single_run_param_recovery(const_model_fittable, model_params=const_model_params)

param_ranges = {"drift": np.linspace(DRIFT_CONST_FIT["minval"], DRIFT_CONST_FIT["maxval"], 8),
                "noise": np.linspace(NOISE_CONST_FIT["minval"], NOISE_CONST_FIT["maxval"], 8),
                "nondectime": np.linspace(OVERLAY_CONST_FIT["minval"], OVERLAY_CONST_FIT["maxval"], 4)}

# %%
fit_results, param_combs, param_names = param_recovery(const_model_fittable, const_model_sim_defaults,
                                                       const_model_factory_dict, param_ranges)
