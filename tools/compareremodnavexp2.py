import pandas as pd
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt

def compare(args):
    filenames = args.filenames
    start_t = args.start_time
    end_t = args.end_time

    dfs = []
    len = 0
    for i, filename in enumerate(filenames):
        df = pd.read_csv(filename, delimiter = '\t')
        df = df[df['onset'] > start_t][df['onset'] + df['duration'] < end_t]
        if i == 0:
            end_idx = df.__len__()
        dfs.append(df)
        len = i + 1

    print(dfs)

    fig, axes = plt.subplots(3, 5, figsize = (24, 12))

    df_dats = pd.DataFrame()
    for idx, df in enumerate(dfs):
        for label in ['FIXA', 'SACC', 'PURS', 'ISAC']:
            df_dats.loc[idx, 'n_' + label] = df['label'].value_counts()[label]
            df_dats.loc[idx, 'f_' + label] = df['label'].value_counts()[label] / (end_t - start_t)
            df_dats.loc[idx, '%_' + label] = df['label'].value_counts(normalize = True)[label]
            df_dats.loc[idx, 'μ_t_' + label] = np.mean(df[df['label'] == label]['duration'])
            df_dats.loc[idx, 'σ_t_' + label] = np.std(df[df['label'] == label]['duration'])
            df_dats.loc[idx, 'μ_v_max_' + label] = np.mean(df[df['label'] == label]['peak_vel'])
            df_dats.loc[idx, 'σ_v_max_' + label] = np.std(df[df['label'] == label]['peak_vel'])
            df_dats.loc[idx, 'μ_v_' + label] = np.average(a = df[df['label'] == label]['avg_vel'], weights = df[df['label'] == label]['duration'])
    
    df_dats.index = filenames
    print(df_dats)

if __name__ == "__main__":
    parser = ap.ArgumentParser(description = "compare classification files")
    parser.add_argument('filenames', nargs='+', help='CSV files to process')
    parser.add_argument('-s', '--start_time', default = 0, help = 'start time (inclusive)')
    parser.add_argument('-e', '--end_time', default = 100, help = 'end time (exclusive)')
    args = parser.parse_args()
    compare(args)

