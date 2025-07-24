import pandas as pd
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.widgets import Slider
from matplotlib import cm

def visualize(args):
    # Params

    video_pixels = (2880, 1360)


    fname = args.filename
    drop_neg = args.drop_negative_timestamp == 'y'

    # load dataframe & clean data
    ## Load dataframe
    df = pd.read_csv(fname)
    if drop_neg:
        df = df[df['relative_to_video_first_frame_timestamp'] >= 0]

    ## Clean rows with bad status
    cols1 = [
        'left_projected_x', 
        'left_projected_y',
        'left_forward_x',   
        'left_forward_y', 
        'left_forward_z'
    ]
    df.loc[df['left_status'] != 2, cols1] = [0, 0, 0, 0, 1]

    cols2 = [
        'right_projected_x', 
        'right_projected_y',
        'right_forward_x',  
        'right_forward_y', 
        'right_forward_z'
    ]
    df.loc[df['right_status'] != 2, cols2] = [0, 0, 0, 0, 1]

    cols3 = [
        'gaze_projected_to_left_view_x',  
        'gaze_projected_to_left_view_y',
        'gaze_projected_to_right_view_x', 
        'gaze_projected_to_right_view_y',
        'gaze_forward_x',                 
        'gaze_forward_y',
        'gaze_forward_z'
    ]
    df.loc[df['status'] != 2, cols3] = [0, 0, 0, 0, 0, 0, 1]

    ## Clean out commas and fix types
    df.map(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)

    cols = cols1 + cols2 + cols3

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    df['datetime'] = pd.to_datetime(df['relative_to_video_first_frame_timestamp'], unit = 'ns')
    df.reset_index(inplace = True, drop = True)
    df.index = pd.DatetimeIndex(df['datetime'])

    if args.clean_method == 'i':
        for col in cols:
            df[col] = df[col].replace(0, np.nan)
            print(df[col].index.value_counts())
            df[col] = df[col].interpolate(method = 'time')
            df.iloc[0][col] = 0

    if args.clean_method == 'r':
        df = df.loc[(df[cols] != 0).all(axis = 1)]

    

    # Build out data
    ## Time / dependent vars
    time_s = df['video_seconds'] = df['relative_to_video_first_frame_timestamp'] / (10 ** 9)
    time_hms = time_s.apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if time.strftime('%H:%M:%S', time.gmtime(x)) != '23:59:59' else 'neg')

    ## Status / stability
    gaze_status = df['status']
    left_status = df['left_status']
    right_status = df['right_status']
    gaze_valid = df['valid'] = df['status'] == 2
    left_valid = df['left_valid'] = df['left_status'] == 2
    right_valid = df['right_valid'] = df['right_status'] == 2
    stability = df['stability']

    ## Eye tracking
    ### Normalized vectors
    left_x = df['left_forward_x']
    left_y = df['left_forward_y']
    left_m = df['left_magnitude'] = np.sqrt(left_x ** 2 + left_y ** 2)

    gaze_x = df['gaze_forward_x']
    gaze_y = df['gaze_forward_y']
    gaze_m = df['gaze_magnitude'] = np.sqrt(gaze_x ** 2 + gaze_y ** 2)

    right_x = df['right_forward_x']
    right_y = df['right_forward_y']
    right_m = df['right_magnitude'] = np.sqrt(right_x ** 2 + right_y ** 2)

    #### Derivatives
    left_Dx = df['left_Dx'] = np.gradient(left_x, time_s)
    left_Dy = df['left_Dy'] = np.gradient(left_y, time_s)
    left_Dm = df['left_Dm'] = np.gradient(left_m, time_s)
    left_mD = df['left_mD'] = np.sqrt(left_Dx ** 2 + left_Dy ** 2)

    gaze_Dx = df['gaze_Dx'] = np.gradient(gaze_x, time_s)
    gaze_Dy = df['gaze_Dy'] = np.gradient(gaze_y, time_s)
    gaze_Dy = df['gaze_Dm'] = np.gradient(gaze_m, time_s)
    gaze_mD = df['gaze_mD'] = np.sqrt(gaze_Dx ** 2 + gaze_Dy ** 2)

    right_Dx = df['right_Dx'] = np.gradient(right_x, time_s)
    right_Dy = df['right_Dy'] = np.gradient(right_y, time_s)
    right_Dm = df['right_Dm'] = np.gradient(right_m, time_s)
    right_mD = df['right_mD'] = np.sqrt(right_Dx ** 2 + right_Dy ** 2)
    
    ### Angle calculations
    left_the = df['left_the'] = np.arctan(df['left_forward_y'] / df['left_forward_z']) * 180 / np.pi
    left_phi = df['left_phi'] = np.arctan(df['left_forward_x'] / df['left_forward_z']) * 180 / np.pi
    left_ang = df['left_ang'] = np.arctan(df['left_magnitude'] / df['left_forward_z']) * 180 / np.pi

    gaze_the = df['gaze_the'] = np.arctan(df['gaze_forward_y'] / df['gaze_forward_z']) * 180 / np.pi
    gaze_phi = df['gaze_phi'] = np.arctan(df['gaze_forward_x'] / df['gaze_forward_z']) * 180 / np.pi
    gaze_ang = df['gaze_ang'] = np.arctan(df['gaze_magnitude'] / df['gaze_forward_z']) * 180 / np.pi

    right_the = df['right_the'] = np.arctan(df['right_forward_y'] / df['right_forward_z']) * 180 / np.pi
    right_phi = df['right_phi'] = np.arctan(df['right_forward_x'] / df['right_forward_z']) * 180 / np.pi
    right_ang = df['right_ang'] = np.arctan(df['right_magnitude'] / df['right_forward_z']) * 180 / np.pi

    #### Derivatives
    left_Dthe = df['left_Dthe'] = np.gradient(left_the, time_s)
    left_Dphi = df['left_Dphi'] = np.gradient(left_phi, time_s)
    left_Dang = df['left_Dang'] = np.gradient(left_ang, time_s)
    left_ang_mD = df['left_ang_mD'] = np.sqrt(left_Dphi ** 2 + left_Dthe ** 2)

    gaze_Dthe = df['gaze_Dthe'] = np.gradient(gaze_the, time_s)
    gaze_Dphi = df['gaze_Dphi'] = np.gradient(gaze_phi, time_s)
    gaze_Dang = df['gaze_Dang'] = np.gradient(gaze_ang, time_s)
    gaze_ang_mD = df['gaze_ang_mD'] = np.sqrt(gaze_Dphi ** 2 + gaze_Dthe ** 2)

    right_Dthe = df['right_Dthe'] = np.gradient(right_the, time_s)
    right_Dphi = df['right_Dphi'] = np.gradient(right_phi, time_s)
    right_Dang = df['right_Dang'] = np.gradient(right_ang, time_s)
    right_ang_mD = df['right_ang_mD'] = np.sqrt(right_Dphi ** 2 + right_Dthe ** 2)

    ### Video coordinates
    left_proj_x = df['left_proj_x'] = 0.25 * video_pixels[0] * (1.0 + df['left_projected_x'])
    left_proj_y = df['left_proj_y'] = 0.5 * video_pixels[1] * (1.0 - df['left_projected_y'])
    left_proj_m = df['left_proj_m'] = np.sqrt(left_proj_x ** 2 + left_proj_y ** 2)

    gaze_proj_left_x = df['gaze_proj_left_x'] = 0.25 * video_pixels[0] * (1.0 + df['gaze_projected_to_left_view_x'])
    gaze_proj_left_y = df['gaze_proj_left_y'] = 0.5 * video_pixels[1] * (1.0 - df['gaze_projected_to_left_view_y'])
    gaze_proj_left_m = df['gaze_proj_left_m'] = np.sqrt(gaze_proj_left_x ** 2 + gaze_proj_left_y ** 2)

    right_proj_x = df['right_proj_x'] = 0.25 * video_pixels[0] * (3.0 + df['right_projected_x'])
    right_proj_y = df['right_proj_y'] = 0.5 * video_pixels[1] * (1.0 - df['right_projected_y'])
    right_proj_m = df['right_proj_m'] = np.sqrt(right_proj_x ** 2 + right_proj_y ** 2)

    gaze_proj_right_x = df['gaze_proj_right_x'] = 0.25 * video_pixels[0] * (3.0 + df['gaze_projected_to_right_view_x'])
    gaze_proj_right_y = df['gaze_proj_right_y'] = 0.5 * video_pixels[1] * (1.0 - df['gaze_projected_to_right_view_y'])
    gaze_proj_right_m = df['gaze_proj_right_m'] = np.sqrt(gaze_proj_right_x ** 2 + gaze_proj_right_y ** 2)

    #### Derivatives
    left_proj_Dx = df['left_proj_Dx'] = np.gradient(left_proj_x, time_s)
    left_proj_Dy = df['left_proj_Dy'] = np.gradient(left_proj_y, time_s)
    left_proj_Dm = df['left_proj_Dm'] = np.gradient(left_proj_m, time_s)
    left_proj_mD = df['left_proj_mD'] = np.sqrt(left_proj_Dx ** 2 + left_proj_Dy ** 2)

    gaze_proj_left_Dx = df['gaze_proj_left_Dx'] = np.gradient(gaze_proj_left_x, time_s)
    gaze_proj_left_Dy = df['gaze_proj_left_Dy'] = np.gradient(gaze_proj_left_y, time_s)
    gaze_proj_left_Dm = df['gaze_proj_left_Dm'] = np.gradient(gaze_proj_left_m, time_s)
    gaze_proj_left_mD = df['gaze_proj_left_mD'] = np.sqrt(gaze_proj_left_Dx ** 2 + gaze_proj_left_Dy ** 2)

    right_proj_Dx = df['right_proj_Dx'] = np.gradient(right_proj_x, time_s)
    right_proj_Dy = df['right_proj_Dy'] = np.gradient(right_proj_y, time_s)
    right_proj_Dm = df['right_proj_Dm'] = np.gradient(right_proj_m, time_s)
    right_proj_mD = df['right_proj_mD'] = np.sqrt(right_proj_Dx ** 2 + right_proj_Dy ** 2)

    gaze_proj_right_Dx = df['gaze_proj_right_Dx'] = np.gradient(gaze_proj_right_x, time_s)
    gaze_proj_right_Dy = df['gaze_proj_right_Dy'] = np.gradient(gaze_proj_right_y, time_s)
    gaze_proj_right_Dm = df['gaze_proj_right_Dm'] = np.gradient(gaze_proj_right_m, time_s)
    gaze_proj_right_mD = df['gaze_proj_right_mD'] = np.sqrt(gaze_proj_right_Dx ** 2 + gaze_proj_right_Dy ** 2)
    
    ## Eye information
    left_dilation = df['left_pupil_diameter_in_mm']
    left_openness = df['left_eye_openness']

    right_dilation = df['right_pupil_diameter_in_mm']
    right_openness = df['right_eye_openness']

    ### Derivatives
    left_Ddil = df['left_Ddil'] = np.gradient(left_dilation, time_s)
    left_Dopen = df['left_Dopen'] = np.gradient(left_openness, time_s)
    
    right_Ddil = df['right_Ddil'] = np.gradient(right_dilation, time_s)
    right_Dopen = df['right_Dopen'] = np.gradient(right_openness, time_s)

    # Set up subplots
    fig, axes = plt.subplots(3, 5, figsize = (24, 12))
    graphs = pd.DataFrame(
        columns = ['title', 'ax_idx', 'graphtype', 'window', 'x_metric', 'y_metric', 'heat_metric', 'status_metric', 'x_label', 'y_label', 'heat_label', 'hover_metrics'],
        data = [   
            ["Gaze Φ"                   , (0, 0)    , 'line'    , None  , time_s        , gaze_phi      , np.abs(gaze_Dphi)     , None          , 'Time (s)'                        , 'Horizontal rotation (°)'         , 'Horizontal angular speed (°/s)'  , [('Time', time_s, 's')    , ('Timestamp', time_hms, '')   , ('Gaze Φ', gaze_phi, '°')                     , ('dΦ/dt', gaze_Dphi, '°/s')                                                                       ]],
            ["Gaze Θ"                   , (1, 0)    , 'line'    , None  , time_s        , gaze_the      , np.abs(gaze_Dthe)     , None          , 'Time (s)'                        , 'Vertical rotation (°)'           , 'Vertical angular speed (°/s)'    , [('Time', time_s, 's')    , ('Timestamp', time_hms, '')   , ('Gaze Θ', gaze_the, '°')                     , ('dΘ/dt', gaze_Dthe, '°/s')                                                                       ]],
            ["Gaze ω"                   , (2, 0)    , 'line'    , None  , time_s        , gaze_ang_mD   , gaze_ang_mD           , None          , 'Time (s)'                        , 'Angular speed (°/s)'             , 'Angular speed (°/s)'             , [('Time', time_s, 's')    , ('Timestamp', time_hms, '')   , ('Gaze ω', gaze_ang_mD, '°/s')                                                                                                                    ]],
            ["Status"                   , (2, 1)    , 'scat'    , None  , time_s        , gaze_status   , None                  , None          , 'Time (s)'                        , 'Angular speed (°/s)'             , 'Angular speed (°/s)'             , [('Time', time_s, 's')    , ('Timestamp', time_hms, '')   , ('Gaze Φ', gaze_phi, '°')                     , ('Gaze Θ', gaze_the, '°')                                                                         ]],
            ["Left Video Coordinates"   , (0, 1)    , 'line'    , 3     , left_proj_x   , left_proj_y   , np.abs(left_proj_mD)  , None          , 'Left X video coordinate (px)'    , 'Left Y video coordinate (px)'    , 'Left coordinate speed (px/s)'    , [('Time', time_s, 's')    , ('Timestamp', time_hms, '')   , ('Left X Coordinate', left_proj_x, 'px')      , ('Left Y Coordinate', left_proj_y, 'px')      , ('Left Coordinate Speed', left_proj_mD, 'px')     ]],
            ["Right Video Coordinates"  , (1, 1)    , 'line'    , 3     , right_proj_x  , right_proj_y  , np.abs(right_proj_mD) , None          , 'Right X video coordinate (px)'   , 'Right Y video coordinate (px)'   , 'Right coordinate speed (px/s)'   , [('Time', time_s, 's')    , ('Timestamp', time_hms, '')   , ('Right X Coordinate', right_proj_x, 'px')    , ('Right Y Coordinate', right_proj_y, 'px')    , ('Right Coordinate Speed', right_proj_mD, 'px')   ]]
        ]
    )


    scs = []
    for _, graph in graphs.iterrows():
        ax = axes[graph['ax_idx']]
        title = graph['title']
        yval = graph['y_metric']
        xval = graph['x_metric']
        hval = graph['heat_metric']
        slider = None
        window = graph['window']

        if not pd.isna(window):
            ax_slider = inset_axes(
                ax,
                width = '100%', height = '5%',
                loc = 'lower center',
                bbox_to_anchor = (0, -0.2, 1, 1),
                bbox_transform = ax.transAxes,
                borderpad = 0
            )

            slider = Slider(
                ax = ax_slider,
                label = "Window start (s)",
                valmin = time_s.min(),
                valmax = time_s.max() - window,
                valinit = time_s.min(),
                valstep = 0.05,
            )
            mask = (time_s > time_s.min()) & (time_s < time_s.min() + window)

            xval = xval[mask]
            yval = yval[mask]
            hval = hval[mask]

        if graph['graphtype'] == 'line':
            pts = np.vstack([xval, yval]).T
            segs = np.stack([pts[:-1], pts[1:]], axis = 1)

            if (hval is not None):
                if graph['status_metric'] is None:
                    norm_hval = Normalize(vmin = np.nanpercentile(hval, 5), vmax = np.nanpercentile(hval, 95))
                    lc = LineCollection(segs, cmap = 'viridis', norm = norm_hval, linewidth = 1.5)
                    fig.colorbar(lc, ax=ax, label = graph['heat_label'])

                    lc.set_array(hval);
                else:
                    norm_hval = Normalize(vmin = np.nanpercentile(hval, 5), vmax = np.nanpercentile(hval, 95))
                    cmap = cm.get_cmap('viridis')
                    heat_rgba = cmap(norm_hval(hval.values[:-1]))
                    valid = (graph['status_metric'][:-1]) & (graph['status_metric'][1:])
                    heat_rgba[~valid] = np.array([1.0, 0.0, 0.0, 1.0])

                    lc = LineCollection(segs, colors = heat_rgba, linewidth = 1.5)

                    sm = cm.ScalarMappable(norm = norm_hval, cmap = cmap)
                    sm.set_array([])
                    fig.colorbar(sm, ax = ax, label = graph['heat_label'])
            else:
                lc = LineCollection(segs, linewidth = 1.5)
                lc.set_array(hval);

            ax.add_collection(lc)
            sc = ax.scatter(xval, yval, s = 15, alpha = 0, picker = False)
            scs.append({'ax' : ax, 'sc' : sc, 'graph' : graph, 'xarr' : xval, 'yarr' : yval, 'harr' : hval, 'slider': slider, 'window': window})
        elif graph['graphtype'] == 'scat':
            sc = ax.scatter(xval, yval, s = 15, alpha = 1, picker = False)
            scs.append({'ax' : ax, 'sc' : sc, 'graph' : graph, 'xarr' : xval, 'yarr' : yval, 'harr' : hval, 'slider': slider, 'window': window})
        ax.set_title(title)
        ax.set_xlim(graph['x_metric'].min() - np.ptp(graph['x_metric']) / 10, graph['x_metric'].max() + np.ptp(graph['x_metric']) / 10)
        ax.set_ylim(graph['y_metric'].min() - np.ptp(graph['y_metric']) / 10, graph['y_metric'].max() + np.ptp(graph['y_metric']) / 10)
        ax.set_xlabel(graph['x_label'])
        ax.set_ylabel(graph['y_label'])

    # Common formatting
    fig.suptitle('Eye Tracking Visualizations for ' + os.path.basename(fname), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.1)

    ## Annotations
    anns = {}
    for elm in scs:
        ax = elm['ax']
        ann = ax.annotate(
            '', xy = (0, 0), xytext = (15, 15),
            textcoords = 'offset points',
            bbox = dict(boxstyle = 'round', fc = 'w', alpha = 1),
            arrowprops = dict(arrowstyle = '->'),
            annotation_clip = False,
            zorder = 10
        )
        ann.set_visible(False)
        anns[ax] = ann
    
    def on_move(event):
        need_redraw = False
        for elm in scs:
            ax = elm['ax']
            graph = elm['graph']
            xarr = elm['xarr']
            yarr = elm['yarr']
            ann = anns[ax]
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:

                disp_pts = ax.transData.transform(
                    np.column_stack((xarr, yarr))
                )
                dx   = disp_pts[:, 0] - event.x
                dy   = disp_pts[:, 1] - event.y
                dist = np.hypot(dx, dy)
                idx  = np.argmin(dist)

                xi, yi = xarr[idx], yarr[idx]
                ann.xy = (xi, yi)
                
                # dx = xarr - event.xdata
                # dy = yarr - event.ydata
                # dist = np.hypot(dx, dy)
                # idx = np.argmin(dist)
                
                # dist, idx = tree.query([event.xdata, event.ydata])
                # xi, yi = xarr[idx], yarr[idx]
                # ann.xy = (xi, yi)

                text = f''
                for hover_metric in graph['hover_metrics']:
                    text += f'{hover_metric[0]}: {str(hover_metric[1][idx])[:10]}{hover_metric[2]}'
                    if hover_metric != graph['hover_metrics'][-1]:
                        text += f'\n'

                ann.set_text(text)
                ann.set_visible(True)
                need_redraw = True
            else:
                if ann.get_visible():
                    ann.set_visible(False)
                    need_redraw = True
        if need_redraw:
            fig.canvas.draw_idle()

    def slider_update(val):
        need_redraw = False

        for elm in scs:
            ax = elm['ax']
            graph = elm['graph']
            xarr = elm['xarr']
            yarr = elm['yarr']
            harr = elm['harr']
            slider = elm['slider']
            window = elm['window']

            if slider is None:
                continue

            t = slider.val
            mask = (time_s >= t) & (time_s < t + window)
            xarr = graph['x_metric'][mask]
            yarr = graph['y_metric'][mask]
            harr = graph['heat_metric'][mask] if graph['heat_metric'] is not None else None

            lc = ax.collections[0]
            pts  = np.vstack([xarr, yarr]).T
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc.set_segments(segs)
            if harr is not None:
                lc.set_array(harr)

            need_redraw = True



        if need_redraw:
            fig.canvas.draw_idle()
    
    for elm in scs:
        if elm['slider'] is not None:
            elm['slider'].on_changed(slider_update)
    

    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()
    
 
if __name__ == "__main__":
    parser = ap.ArgumentParser(description = "visualize recorded eye tracking")
    parser.add_argument('filename', help='CSV file to process')
    parser.add_argument('-n', '--drop_negative_timestamp', required = False, default = 'y', choices = ['y', 'n'], help='include negative timestamps prior to video start?')
    parser.add_argument('-c', '--clean_method', required = False, default = 'i', choices = ['i', 'o', 'a', 'r'], help = 'method for cleaning values, i - interpolate by time, a - moving average, o - 0 fill, r - remove')
    args = parser.parse_args()
    visualize(args)

