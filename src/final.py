import pandas as pd
import numpy as np
import remodnav as rm
import os 
import shutil as su

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
            df_dats.loc[idx, 'μ_v_max_' + type] = np.mean(df[df['label'].isin(labels)]['peak_vel'])
            df_dats.loc[idx, 'σ_v_max_' + type] = np.std(df[df['label'].isin(labels)]['peak_vel'])
            df_dats.loc[idx, 'μ_v_' + type] = np.average(a = df[df['label'].isin(labels)]['avg_vel'], weights = df[df['label'].isin(labels)]['duration'])
    
    df_dats.index = filenames

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

def process_all_data(data_path: str, workspace: str, height: int = 2740, width: int = 2880, PPD: int = 35, savgol_length: float = 0.023, filter_length: float = 0.005):
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

    for subject_name in subject_names:
        subject_path = os.path.join(data_path, subdir)

        subject_data = []
        baseline = get_baseline(subject_path)

        subject_data.append(baseline)
        
        for mode_csv in ['DVE.csv', 'DVEH.csv', 'GVE.csv', 'GVEH.csv']:
            trials_data = get_trials_from_mode(subject_path, mode_csv)

            subject_data.append(trials_data)

        data.append(subject_data)

    print(data)

    for i, subject_data in enumerate(data):
        for j, mode_data in enumerate(subject_data):
            if j == 0:
                full_run_remodnav(mode_data, os.path.join(workspace, subject_names[i] + '_baseline_rmexp.csv'), height, width, PPD, savgol_length, filter_length)
                continue
            else:
                for k, trial_data in enumerate(mode_data):
                    full_run_remodnav(trial_data, os.path.join(workspace, subject_names[i] + '_' + modes[j - 1] + '_' + str(k) + '_rmexp.csv'), height, width, PPD, savgol_length, filter_length)
                
