import pandas as pd
import numpy as np
import remodnav as rm
import os 
import shutil as su
import re
import seaborn as sns
import matplotlib.pyplot as plt

def prep_remodnav_df(raw: pd.DataFrame, height: int, width: int, debug1: str = None, debug2: str = None, outfile: str = None):
    # height: 2740 px
    # width: 2880 px

    df = raw[['status', 'relative_to_video_first_frame_timestamp', 'relative_to_unix_epoch_timestamp', 'gaze_projected_to_left_view_x', 'gaze_projected_to_left_view_y']].reset_index(drop = True)
    df.columns = ['status', 'timestamp', 'unix', 'x_norm', 'y_norm']

    df['x_px'] = df.apply(lambda x: width * 0.5 + width * 0.5 * float(x['x_norm']) if x['status'] == 2 else pd.NA, axis = 1)
    df['y_px'] = df.apply(lambda x: height * 0.5 - height * 0.5 * float(x['y_norm']) if x['status'] == 2 else pd.NA, axis = 1)

    df = df[['timestamp', 'unix', 'x_px', 'y_px', 'status']]
    
    if debug1:
        df.to_csv(debug1, index = False)
    
    dt = int(df['timestamp'].diff().dropna().mean() / 2) 
    start = df['timestamp'].iloc[0]
    end = df['timestamp'].iloc[-1]

    ts = df['timestamp']
    xs = df['x_px']
    ys = df['y_px']
    ss = df['status']

    tsu = np.arange(start, end, dt)
    xsu = []
    ysu = []

    for t in tsu:
        ir = np.searchsorted(ts, t)
        il = ir - 1
        
        if ir >= len(ts) or il < 0:
            xsu.append(pd.NA)
            ysu.append(pd.NA)
            continue
        
        tl, tr, xl, xr, yl, yr, sl, sr = ts[il], ts[ir], xs[il], xs[ir], ys[il], ys[ir], ss[il], ss[ir]

        if (not sl) or (not sr):
            xsu.append(pd.NA)
            ysu.append(pd.NA)
        else:
            f = (t - tl) / dt
            xsu.append(xl * (1 - f) + xr * f)
            ysu.append(yl * (1 - f) + yr * f)

    dfu = pd.DataFrame({'t': tsu, 'x': xsu, 'y': ysu})
    # dfu = dfu.loc[dfu['t'] > 0].reset_index(drop = True)

    if debug2:
        dfu.to_csv(debug2, index = False)
        
    dfu = dfu[['x', 'y']]

    if outfile:
        dfu.to_csv(outfile, index = False)
    
    return dfu, dt


def run_remodnav(infile, outfile, sampling_rate, px2deg, savgol_length, filter_length):
    # px2deg = 1/PPD
    # sampling rate = 1/timestep (timestep is output by prep_remodnav_df function)
    rm.main(args = ['remodnav', infile, outfile, str(px2deg), str(sampling_rate), '--savgol-length', str(savgol_length), '--median-filter-length', str(filter_length)])

def full_run_remodnav(raw, outfile, height, width, PPD, savgol_length, filter_length):
    _, dt = prep_remodnav_df(raw = raw, height = height, width = width, outfile = outfile)
    samp_rate = 1.0 / (dt / 1000000000.0)
    px2deg = 1 / PPD
    run_remodnav(infile = outfile, outfile = outfile, sampling_rate = samp_rate, px2deg = px2deg, savgol_length = savgol_length, filter_length = filter_length)

def remodnav_feature_extract(filenames):
    dfs = []
    lens = []
    for i, filename in enumerate(filenames):
        df = pd.read_csv(filename, delimiter = '\t')
        dfs.append(df)
        lens.append(df.loc[df.__len__() - 1, 'onset'] + df.loc[df.__len__() - 1, 'duration'])


    df_dats = pd.DataFrame()
    for idx, df in enumerate(dfs):
        for type, labels in {'fixations' : ['FIXA'], 'pursuits': ['PURS'], 'saccades': ['SACC', 'ISAC'], 'pso/glissades': ['HPSO', 'LPSO', 'ILPS', 'IHPS']}.items():
            df_dats.loc[idx, 'f_' + type] = 0.0
            df_dats.loc[idx, '%_' + type] = 0.0

            for label in labels:
                if label in df['label'].unique():
                    df_dats.loc[idx, 'f_' + type] += df['label'].value_counts()[label] / lens[idx]
                    df_dats.loc[idx, '%_' + type] += df['label'].value_counts(normalize = True)[label]

            df_dats.loc[idx, 'μ_t_' + type] = np.mean(df[df['label'].isin(labels)]['duration'])
            df_dats.loc[idx, 'σ_t_' + type] = np.std(df[df['label'].isin(labels)]['duration'])
            df_dats.loc[idx, 'μ_vmax_' + type] = np.mean(df[df['label'].isin(labels)]['peak_vel'])
            df_dats.loc[idx, 'σ_vmax_' + type] = np.std(df[df['label'].isin(labels)]['peak_vel'])
            df_dats.loc[idx, 'μ_v_' + type] = np.average(a = df[df['label'].isin(labels)]['avg_vel'], weights = df[df['label'].isin(labels)]['duration'])
    
    df_dats.index = filenames
    return df_dats

def get_trials_from_mode(subject_path: str, mode_csv: str):
    timestamps_path = os.path.join(subject_path, 'processed_timestamps', mode_csv)
    data_path = os.path.join(subject_path, 'Varjo', mode_csv)

    timestamps = pd.read_csv(timestamps_path, header = None)
    df = pd.read_csv(data_path)

    # convert timestamps to Unix time. 

    for column in timestamps.columns:
        timestamps[column] = pd.to_datetime(timestamps[column], format = '%Y-%m-%d %H:%M:%S.%f')
    timestamps = timestamps.map(lambda x: (x.tz_localize('US/Eastern').tz_convert('UTC').timestamp()) * 1000000000)

    trials = []
    for column in timestamps.columns:
        trial_ts = timestamps[column]
        trials.append(df[df['relative_to_unix_epoch_timestamp'] > trial_ts[0]][df['relative_to_unix_epoch_timestamp'] < trial_ts[1]])

    return trials


def get_baseline(subject_path: str):
    baseline_path = os.path.join(subject_path, 'Varjo', 'baseline.csv')
    baseline = pd.read_csv(baseline_path)

    return baseline

def get_pupillometry(data: pd.DataFrame):
    df = pd.DataFrame({'average_openness' : [0.0], 'average_pupil_diameter' : [0.0]})
    left_average_openness = data['left_eye_openness'].mean(skipna = True)
    right_average_openness = data['right_eye_openness'].mean(skipna = True)
    left_average_diameter = data['left_pupil_diameter_in_mm'].mean(skipna = True)
    right_average_diameter = data['right_pupil_diameter_in_mm'].mean(skipna = True)

    df.loc[0, 'average_openness'] = (left_average_openness + right_average_openness) / 2.0
    df.loc[0, 'average_pupil_diameter'] = (left_average_diameter + right_average_diameter) / 2.0
    return df


def process_all_data(data_path: str, workspace: str, workload_path: str, height: int = 2740, width: int = 2880, PPD: int = 35, savgol_length: float = 0.023, filter_length: float = 0.005, bedford_encoding_threshold: int = 6):
    subject_names = []
    modes = ['DVE', 'DVEH', 'GVE', 'GVEH']

    for subdir in os.listdir(data_path):
        subject_names.append(subdir)

    if os.path.exists(workspace):
        su.rmtree(workspace)
        os.mkdir(workspace)
    else:
        os.mkdir(workspace)

    data = []
    wrkld = pd.read_excel(workload_path)
    wrkld.index = wrkld['Subject']
    wrkld.drop('Subject', axis = 1, inplace = True)

    for subject_name in subject_names:
        subject_path = os.path.join(data_path, subject_name)

        subject_data = []
        baseline = get_baseline(subject_path)

        subject_data.append(baseline)
        
        for mode_csv in ['DVE.csv', 'DVEH.csv', 'GVE.csv', 'GVEH.csv']:
            trials_data = get_trials_from_mode(subject_path, mode_csv)

            subject_data.append(trials_data)

        data.append(subject_data)

    for i, subject_data in enumerate(data):
        for j, mode_data in enumerate(subject_data):
            if j == 0:
                full_run_remodnav(mode_data, os.path.join(workspace, subject_names[i] + '_baseline_rmexp.csv'), height, width, PPD, savgol_length, filter_length)
                df = get_pupillometry(mode_data)
                df['bedford_workload'] = [0]
                df['bedford_encoding'] = [0]
                df.to_csv(os.path.join(workspace, subject_names[i] + '_baseline_exexp.csv'), index = False)
                continue
            else:
                for k, trial_data in enumerate(mode_data):
                    full_run_remodnav(trial_data, os.path.join(workspace, subject_names[i] + '_' + modes[j - 1] + '_' + str(k) + '_rmexp.csv'), height, width, PPD, savgol_length, filter_length)
                    df = get_pupillometry(trial_data)
                    df.loc[0, 'bedford_workload'] = wrkld.loc[subject_names[i], modes[j - 1]]
                    df['bedford_encoding'] = df['bedford_workload'].apply(lambda x: 1 if x < bedford_encoding_threshold else 2)
                    df.to_csv(os.path.join(workspace, subject_names[i] + '_' + modes[j - 1] + '_' + str(k) + '_exexp.csv'), index = False)

def load_export(workspace: str):
    allfiles = os.listdir(workspace)

    rmfilenames = []
    exfilenames = []

    for file in allfiles:
        if file.endswith('rmexp.csv'):
            rmfilenames.append(os.path.join(workspace, file))
        elif file.endswith('.csv'):
            exfilenames.append(os.path.join(workspace, file))
        else:
            os.remove(os.path.join(workspace, file))
    
    df = remodnav_feature_extract(rmfilenames)

    df['μ_openness'] = 0.0
    df['μ_diameter'] = 0.0
    df['bedford'] = 0
    df['bedford_enc'] = 0

    for file in exfilenames:
        indexpath = file.replace('exexp', 'rmexp')
        ex = pd.read_csv(file)
        ex.columns = ['μ_openness', 'μ_diameter', 'bedford', 'bedford_enc']
        for col in ['μ_openness', 'μ_diameter', 'bedford', 'bedford_enc']:
            df.loc[indexpath, col] = ex.loc[0, col]
    df['Subject'] = df.index
    df['Mode'] = df['Subject'].apply(lambda x: re.search(r'_(baseline|DVEH|DVE|GVEH|GVE)', x).group(1))
    df['Subject'] = df['Subject'].apply(lambda x: re.search(r'Subject_(\d+)', x).group(1))
    df.reset_index(inplace = True, drop = True)
    return df

def baseline_normalize(df: pd.DataFrame):

    baseline = df[df['Mode'] == 'baseline'].set_index('Subject', drop = True)
    exclude_cols = ['bedford', 'bedford_enc', 'Subject', 'Mode']
    include_cols = df.columns.difference(exclude_cols)
    
    normalized_values = df.apply(lambda row: (row[include_cols] - baseline.loc[row["Subject"], include_cols]) / baseline.loc[row["Subject"], include_cols], axis=1)
    df_normalized = pd.concat([df[exclude_cols], normalized_values], axis=1)

    return df_normalized[df_normalized['Mode'] != 'baseline']
    
def average_trials(df: pd.DataFrame):
    df_averaged = df.groupby(['Subject', 'Mode']).mean().reset_index()

    return df_averaged

def z_score_standardize(df: pd.DataFrame):
    include_cols = df.columns.difference(['Subject', 'Mode', 'bedford', 'bedford_enc'])

    df_zscored = df.copy()
    df_zscored[include_cols] = df.groupby('Mode')[include_cols].transform(
        lambda x: (x - x.mean() / x.std(ddof = 0))
    )

    return df_zscored


def analyze_all_data(workspace: str, func1 = lambda x: x, func2 = lambda x: x, func3 = lambda x: x):
    df = load_export(workspace)

    df = func1(df)

    df = func2(df)

    df = func3(df)

    print(df)
    return df

def plotter(df: pd.DataFrame):
    sns.set(style = 'whitegrid')

    include_cols = df.columns.difference(['Subject', 'Mode', 'bedford', 'bedford_enc'])

    n_cols = 4
    n_rows = -(-len(include_cols) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(include_cols):
        sns.boxplot(data=df, x='bedford_enc', y=col, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col} by bedford_enc')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plot_path = "out.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    plot_path


df = analyze_all_data('wrk', baseline_normalize, z_score_standardize)
print(df['bedford_enc'].value_counts())
plotter(df)