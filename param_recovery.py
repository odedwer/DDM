## ConstModel param recovery
# Fittables are - drift rate (constant), noise (constant), overlay (constant)
# Constants (not fittable) - boundaries, starting point (0)

import typing
from itertools import product

import numpy as np
from ddm import Model, Fittable, Solution
from ddm.functions import fit_adjust_model
from ddm.models import LossRobustBIC, DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision

DEFAULT_N_SAMPLES = 1000


def sample_model(model, n_samples=DEFAULT_N_SAMPLES):
    sol: Solution = model.solve()
    samples = sol.resample(n_samples)
    return samples


def fit_model(model, **kwargs):
    if "samples" not in kwargs.keys():
        kwargs["samples"] = sample_model(model) if "n_samples" not in kwargs.keys() else sample_model(model, kwargs[
            "n_samples"])
    fit_adjust_model(kwargs["samples"], model,
                     fitting_method="differential_evolution",
                     lossfunction=LossRobustBIC)


def param_recovery(fittable_model: Model, param_defaults_dict: dict, param_factory_dict: dict, param_ranges: dict):
    sorted_ranges = np.array(list(param_ranges.values()))[np.argsort(list(param_ranges.keys()))]
    param_combs = list(product(*sorted_ranges))
    model_params = param_defaults_dict.copy()
    param_names = list(param_ranges.keys())
    fit_results = []
    for param_comb in param_combs:
        for i, param_name in enumerate(param_names):
            model_params[param_name] = param_factory_dict[param_name](param_comb[i])
        fitted_parameters = single_run_param_recovery(fittable_model, model_params)
        # TODO: Add appending of fit results and compute MSE \\ plot the fitted vs simulated


def single_run_param_recovery(fittable_model, model_params, n_samples=DEFAULT_N_SAMPLES) -> typing.List:
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
    fitted_params = list(map(lambda a: float(a), fittable_model.get_model_parameters()))
    return sorted(list(zip(fittable_model.get_model_parameter_names(), fitted_params)))


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
const_model_params = {"drift": DriftConstant(drift=2), "noise": NoiseConstant(noise=1.5),
                      "bound": BoundConstant(B=BOUND), "overlay": OverlayNonDecision(nondectime=.3)}

const_model_factory_dict = {"drift": DriftConstant, "noise": NoiseConstant, "overlay": OverlayNonDecision}

const_model_params.update(model_sim_defaults)

params = single_run_param_recovery(const_model_fittable, model_params=const_model_params)
