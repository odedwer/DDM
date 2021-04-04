import pandas as pd
import os
import numpy as np

# %%
FINAL_FILENAME = 'PVI subjects for ddm.csv'
FILE_PATH = r"S:\Lab-Shared\Experiments\PVI - Priming Intentions\PVI_EEG_junior_100hz\results\edat"
EXCLUDED_FILES = [r"PVI_EEGjunior-100hz-behavioral Analysis", r"EMG results","RT_instructed_correct_error"]
WANTED_COLUMNS = ['Subject', 'Trial', 'PrimeCanvas', 'TargetCanvas', 'ActualResp', 'ResponseTime']
COLUMN_DICT = {'PrimeCanvas': 'Prime', 'TargetCanvas': 'Cue', 'ResponseTime': "RT", "ActualResp": "Resp"}
CANVAS_DICT = {1: 'L', 2: 'R', 3: 'FC', 4: 'N'}
RESPONSE_DICT = {"z": "L", r"{/}": "R", np.nan: "None"}


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
    df['Trial'] = df.index + 1
    df = df[WANTED_COLUMNS]
    df.rename(columns=COLUMN_DICT, inplace=True)
    df['Resp'].replace(RESPONSE_DICT, inplace=True)
    df[['Prime', 'Cue']] = df[['Prime', 'Cue']].replace(CANVAS_DICT)
    df = df[df['RT'].notna()]
    df = df[df['Subject'].notna()]
    df['Subject'] = df['Subject'].astype('int')
    df.to_csv(FINAL_FILENAME)
