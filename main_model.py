import pandas as pd
import numpy as np
import ddm
import matplotlib.pyplot as plt
import seaborn as sns
from generate_csv_from_edat_excel import FINAL_FILENAME

# %% basic raw distributions
# RT over
df = pd.read_csv(FINAL_FILENAME, index_col=0)
sub1 = df.loc[df["subject"] == 223,]
sns.displot(sub1, x="rt", col="cue", row="prime", hue="resp", kind="kde",rug=False)
sns.displot(df, x="rt", col="cue", row="prime", hue="resp", kind="kde")
# signal detection statistics per subject
subjectwise_signal_detection = df.groupby('subject').agg(
    no_resp=pd.NamedAgg(column='resp', aggfunc=lambda col: (col == "none").sum() / col.size)
)
