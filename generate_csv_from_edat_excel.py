import pandas as pd
import os
import numpy as np

# %%
FINAL_FILENAME = 'PVI senior subjects for ddm.csv'
FILE_PATH = r"S:\Lab-Shared\Experiments\PVI - Priming Intentions\PVI_EEG_senior_60hz\results\edat"
EXCLUDED_FILES = [r"PVI_EEGjunior-100hz-behavioral Analysis", r"EMG results","RT_instructed_correct_error","PVI_EEGsenior_60hz - Behavioral analysis"]
WANTED_COLUMNS = ['Subject', 'trial', 'PrimeCanvas', 'TargetCanvas', 'ActualResp', 'ResponseTime']
COLUMN_DICT = {'Subject':'subject','PrimeCanvas': 'prime', 'TargetCanvas': 'cue', 'ResponseTime': "rt", "ActualResp": "resp"}
CANVAS_DICT = {1: 'l', 2: 'r', 3: 'fc', 4: 'n'}
RESPONSE_DICT = {"z": "l", r"{/}": "r", np.nan: "none"}


def is_results_filename(filename):
    return (filename[-4:] == 'xlsx' or filename[-3:] == 'xls') and (filename[:-5] not in EXCLUDED_FILES)


if __name__ == "_main__":
    # load filenames
    filenames = [filename for filename in os.listdir(FILE_PATH) if
                 is_results_filename(filename) and (filename[:-5] not in EXCLUDED_FILES)]
    # read as pandas data frames
    dataframes = [pd.read_excel(os.path.join(FILE_PATH, filename)) for filename in filenames]
    # concatenate to single dataframe and re-format
    df = pd.concat(dataframes,sort=False)
    df['trial'] = df.index + 1
    df = df[WANTED_COLUMNS]
    df.rename(columns=COLUMN_DICT, inplace=True)
    df['resp'].replace(RESPONSE_DICT, inplace=True)
    df[['prime', 'cue']] = df[['prime', 'cue']].replace(CANVAS_DICT)
    df = df[df['subject'].notna()]
    df['subject'] = df['subject'].astype('int')
    df.to_csv(FINAL_FILENAME)
