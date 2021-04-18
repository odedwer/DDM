import pandas as pd
import numpy as np
import ddm
import matplotlib.pyplot as plt
import seaborn as sns
from generate_csv_from_edat_excel import FINAL_FILENAME


@ddm.paranoid.decorators.paranoidclass
class BoundCollapsingExponentialFix(ddm.Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes two parameters:

    - `B` - the bound at time t = 0.
    - `tau` - one divided by the time constant for the collapse,
      should be greater than zero for collapsing bounds, less than
      zero for increasing bounds.  0 gives constant bounds.

    Example usage:

      | bound = BoundCollapsingExponential(B=1, tau=2.1) # Collapsing with time constant 1/2.1
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau"]

    @staticmethod
    def _test(v):
        assert v.B in ddm.paranoid.Positive()

    @staticmethod
    def _generate():
        yield ddm.BoundCollapsingExponential(B=1, tau=1)
        yield ddm.BoundCollapsingExponential(B=.1, tau=.001)
        yield ddm.BoundCollapsingExponential(B=100, tau=100)

    @ddm.paranoid.decorators.accepts(ddm.paranoid.types.base.Self, ddm.paranoid.types.numeric.Number)
    @ddm.paranoid.decorators.returns(ddm.paranoid.types.numeric.Positive0)
    def get_bound(self, t, *args, **kwargs):
        return self.B * np.exp(-self.tau * t)


# %% basic raw distributions
# RT over
# df = pd.read_csv(FINAL_FILENAME, index_col=0)
# sns.displot(df, x="rt", col="cue", row="prime", hue="resp", kind="kde")
# # signal detection statistics per subject
# subjectwise_signal_detection = df.groupby('subject').agg(
#     no_resp=pd.NamedAgg(column='resp', aggfunc=lambda col: (col == "none").sum() / col.size)
# )
# %%
fig = plt.figure()
axs = [fig.subplots(1, 1)]
# %% basic DDM models
DRIFT = 2.2
NOISE = 1.5
BOUND = 3.1
OVERLAY = .1
DX = .001
DT = .01  # sec
DUR = 1  # sec
HW = .03
TAU = 3

ax_idx = 0
m = ddm.Model(name='Simple model', IC=ddm.ICRange(sz=0.4),
              drift=ddm.DriftConstant(drift=1.5, t=3, x=0.4),
              noise=ddm.NoiseConstant(noise=1.5),
              bound=BoundCollapsingExponentialFix(B=1.1, tau=1),
              overlay=ddm.OverlayNonDecisionUniform(nondectime=.1, halfwidth=0.02),
              dx=.001, dt=.01, T_dur=2)
sol: ddm.Solution = m.solve()

axs[ax_idx].plot(sol.model.t_domain(), sol.pdf_err(), label="error")
axs[ax_idx].plot(sol.model.t_domain(), sol.pdf_corr(), label="correct")
axs[ax_idx].legend()
# %%
