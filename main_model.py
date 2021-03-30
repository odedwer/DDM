import pandas as pd
import numpy as np
import ddm
import matplotlib.pyplot as plt
import seaborn as sns
from generate_csv_from_edat_excel import FINAL_FILENAME

# %% basic raw distributions
df = pd.read_csv(FINAL_FILENAME)
sub101 = df.loc[df["Subject"] == 101,]
sns.displot(sub101, x="RT", col="Cue", row="Prime", hue="Response")
