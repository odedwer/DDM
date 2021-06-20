## ConstModel param recovery
# Fittables are - drift rate (constant), noise (constant), overlay (constant)
# Constants (not fittable) - boundaries, starting point (0)

import typing
from itertools import product, combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ddm
from ddm import Model, Fittable, Solution
from ddm.functions import fit_adjust_model
from ddm.models import LossRobustBIC, DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision
from tqdm import tqdm
import pandas as pd
import plotting_functions as pl
from importlib import reload
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
    """

    :param model:
    :param kwargs:
    :return:
    """
    if "samples" not in kwargs.keys():
        kwargs["samples"] = sample_model(model) if "n_samples" not in kwargs.keys() else sample_model(model, kwargs[
            "n_samples"])
    fit_adjust_model(kwargs["samples"], model,
                     fitting_method="differential_evolution",
                     lossfunction=LossRobustBIC, verbose=False)


def param_recovery(fittable_model: Model, param_defaults_dict: dict, param_factory_dict: dict, param_ranges: dict,
                   n_iter=10) -> \
        typing.Tuple[np.array, np.array, list, plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    sorted_ranges = np.array(list(param_ranges.values()))[np.argsort(list(param_ranges.keys()))]
    param_combs = np.array(list(product(*sorted_ranges)))
    model_params = param_defaults_dict.copy()
    fit_param_names = list(param_ranges.keys())
    model_param_names = list(param_factory_dict.keys())
    fit_results = np.full(param_combs.shape + (n_iter,), fill_value=np.nan)
    for i, param_comb in enumerate(tqdm(param_combs)):
        for j, param_name in enumerate(fit_param_names):
            exec('model_params[model_param_names[j]] = param_factory_dict[model_param_names[j]](%s = %f)' % (
                param_name, param_comb[j]))
        for iter in range(n_iter):
            _, fit_results[i, :, iter] = single_run_param_recovery(fittable_model, model_params)
    # unfold to 2D
    param_combs = np.repeat(param_combs, n_iter).reshape(param_combs.shape + (n_iter,))
    fit_results = np.hstack(fit_results).T
    param_combs = np.hstack(param_combs).T
    # plotting simulated vs recovered parameters
    fig_1d_dots = plot_real_vs_recovered(fit_param_names, fit_results, param_combs, param_ranges)
    fig_1d_errbars = plot_real_vs_recovered_error_bars(fit_param_names, fit_results, param_combs, param_ranges)
    fig_interactions = plot_real_vs_recovered_interactions(fit_param_names, fit_results, param_combs, param_ranges)
    # plotting recovery success of parameter interactions
    fig_2d = plot_2d_recovery_loss(fit_param_names, fit_results, param_combs, param_ranges)
    return fit_results, param_combs, fit_param_names, fig_1d_dots, fig_1d_errbars, fig_2d, fig_interactions


def plot_2d_recovery_loss(fit_param_names, fit_results, param_combs, param_ranges,
                          loss=lambda real, fitted: (real - fitted) ** 2, zscore=True):
    if zscore:
        means = param_combs.mean(0)
        stds = param_combs.std(0)
        z_scored_param_combs = (param_combs - means) / stds
        z_scored_fit_results = (fit_results - means) / stds
    else:
        z_scored_param_combs = param_combs
        z_scored_fit_results = fit_results

    loss_mat = loss(z_scored_param_combs, z_scored_fit_results)
    idx_combs = list(combinations(range(len(fit_param_names)), 2))

    # for each plot we want to average all groups by pairs of parameters, easy with pandas group_by function
    df = pd.DataFrame(data=np.hstack([z_scored_param_combs, z_scored_fit_results, loss_mat]),
                      columns=fit_param_names + ["fitted_" + s for s in fit_param_names] + ["loss_" + s for s in
                                                                                            fit_param_names])

    # initiating figure
    fig = plt.figure()
    nrows = np.floor(np.sqrt(len(idx_combs))).astype(int)
    ncols = np.ceil(len(idx_combs) / nrows).astype(int)
    axes = np.array(fig.subplots(nrows, ncols))
    axes = axes.flatten()
    for i, idx in enumerate(idx_combs):
        idx = np.array(idx)
        grouped_df: pd.DataFrame = df.groupby(by=[fit_param_names[idx[0]], fit_param_names[idx[1]]]).mean()
        mse = grouped_df.iloc[:, (2 * len(fit_param_names)) + idx - 2].to_numpy()  # -2 because we grouped
        mse = np.sqrt(mse).mean(1)
        mse = mse.reshape((param_ranges[fit_param_names[idx[0]]].size, param_ranges[fit_param_names[idx[1]]].size))
        # plot
        ax: plt.Axes = axes[i]
        ax.set_title(f"{fit_param_names[idx[0]]} VS {fit_param_names[idx[1]]}")
        ax.set_xlabel(fit_param_names[idx[0]])
        ax.set_ylabel(fit_param_names[idx[1]])
        im = ax.imshow(mse, cmap=plt.cm.jet)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_real_vs_recovered(ctor_param_names, fit_results, param_combs, param_ranges) -> typing.Tuple[
    plt.Figure, plt.Axes]:
    """
    Plots the recovered parameters vs the real parameters used for simulation - each simulation single dot.
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
        ax.plot(plot_lim, plot_lim, '--', color='gray')
    fig.tight_layout()
    return fig


def plot_real_vs_recovered_error_bars(ctor_param_names, fit_results, param_combs, param_ranges) -> typing.Tuple[
    plt.Figure, plt.Axes]:
    """
    Plots the recovered parameters vs the real parameters used for simulation - mean and error bars
    (currently - standard error).
    :param ctor_param_names: Names of the recovered parameters
    :param fit_results: the recovered parameters
    :param param_combs: the real parameters used for simulation
    :param param_ranges: the range of parameters used for simulation
    :return: figure & axes
    """
    df = pd.DataFrame(data=np.hstack([param_combs, fit_results]),
                      columns=ctor_param_names + ["fitted_" + s for s in ctor_param_names])
    fig = plt.figure()
    for i, param_name in enumerate(ctor_param_names):
        ax: plt.Axes = fig.add_subplot(1, len(ctor_param_names), i + 1)
        grouped_df: pd.DataFrameGroupBy = df.groupby(by=param_name)
        fit_results_mean = grouped_df.mean()["fitted_" + param_name]
        fit_results_err = grouped_df.std()["fitted_" + param_name]
        ax.errorbar(np.unique(param_ranges[param_name]), fit_results_mean, yerr=fit_results_err)
        ax.set_title(param_name)
        plot_lim = np.linspace(param_ranges[param_name].min(), param_ranges[param_name].max(), 2)
        ax.plot(plot_lim, plot_lim, '--', color='gray')
    fig.tight_layout()
    return fig


def plot_real_vs_recovered_interactions(ctor_param_names, fit_results, param_combs, param_ranges) -> typing.Tuple[
    plt.Figure, plt.Axes]:
    """
    Plots the recovered parameters vs the real parameters used for simulation
    for each param we plot mean and error bars according to the level of each of the other params
    (currently - errorbars = standard error).
    :param ctor_param_names: Names of the recovered parameters
    :param fit_results: the recovered parameters
    :param param_combs: the real parameters used for simulation
    :param param_ranges: the range of parameters used for simulation
    :return: figure & axes
    """
    n_params = len(ctor_param_names)
    df = pd.DataFrame(data=np.hstack([param_combs, fit_results]),
                      columns=ctor_param_names + ["fitted_" + s for s in ctor_param_names])
    fig = plt.figure()
    for i, param_name_main in enumerate(ctor_param_names):
        for j, param_name_secondary in enumerate(ctor_param_names):
            ax: plt.Axes = fig.add_subplot(n_params, n_params, i * n_params + j + 1)
            # TODO: remove the axes for the cases of i==j, but leave the title & ylabel
            if i == 0:
                ax.set_title(param_name_secondary)
            if j == 0:
                ax.set_ylabel(param_name_main, fontsize=rcParams['axes.titlesize'])

            if i == j:
                continue
            ax: plt.Axes = fig.add_subplot(n_params, n_params, i * n_params + j + 1)
            grouped_df: pd.DataFrameGroupBy = df.groupby(by=[param_name_main, param_name_secondary])
            fit_results_mean = grouped_df.mean()["fitted_" + param_name_main].to_numpy()
            fit_results_err = grouped_df.std()["fitted_" + param_name_main].to_numpy()

            fit_results_mean = fit_results_mean.reshape((param_ranges[param_name_main].size,
                                                         param_ranges[param_name_secondary].size))
            fit_results_err = fit_results_err.reshape((param_ranges[param_name_main].size,
                                                       param_ranges[param_name_secondary].size))
            for idx, val in enumerate(param_ranges[param_name_secondary]):
                ax.errorbar(np.unique(param_ranges[param_name_main]), fit_results_mean[:, idx],
                            yerr=fit_results_err[:, idx], label=val)

            ax.legend()
            plot_lim = np.linspace(param_ranges[param_name_main].min(), param_ranges[param_name_main].max(), 2)
            ax.plot(plot_lim, plot_lim, '--', color='gray')
    fig.tight_layout()
    return fig


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

param_ranges = {"drift": np.linspace(DRIFT_CONST_FIT["minval"], DRIFT_CONST_FIT["maxval"], 2),
                "noise": np.linspace(NOISE_CONST_FIT["minval"], NOISE_CONST_FIT["maxval"], 2),
                "nondectime": np.linspace(OVERLAY_CONST_FIT["minval"], OVERLAY_CONST_FIT["maxval"], 2)}

# %%
fit_results, param_combs, fit_param_names, fig_1d_dots, fig_1d_errbars, fig_2d, fig_interactions = param_recovery(
    const_model_fittable, const_model_sim_defaults,
    const_model_factory_dict, param_ranges, n_iter=2)

plt.show()