import pandas as pd
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize(fname):
    df = pd.read_csv(fname)

    df['video_seconds'] = df['relative_to_video_first_frame_timestamp'] / (10 ** 9)
    df['gaze_projected_to_right_view_x'] = df["gaze_projected_to_right_view_x"].apply(lambda x: x if isinstance(x, (int, float)) else 0)
    df['gaze_projected_to_right_view_y'] = df["gaze_projected_to_right_view_y"].apply(lambda x: x if isinstance(x, (int, float)) else 0)
    df['proj_gaze_mag_left'] = np.sqrt(df['gaze_projected_to_right_view_x'] ** 2 + df['gaze_projected_to_right_view_y'] ** 2)

    print(df["video_seconds"])
    
    plt.figure(figsize=(10, 6))

    plt.subplot2grid(shape = (3, 3), loc = (1, 1))

    invalid = df[df['status'] == 0]
    plt.plot(
        invalid["video_seconds"], 
        invalid["proj_gaze_mag_left"], 
        label= "Eyes not found", 
        alpha=0.8,
    )

    semivalid = df[df['status'] == 1]
    plt.plot(
        semivalid["video_seconds"], 
        semivalid["proj_gaze_mag_left"], 
        label= "Not yet calibrated", 
        alpha=0.8,
    )

    valid = df[df['status'] == 2]
    plt.plot(
        valid["video_seconds"], 
        valid["proj_gaze_mag_left"], 
        label= "Tracking online", 
        alpha=0.8,
    )

    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("2D Gaze Magnitude (normalized)")
    plt.title("Gaze Magnitude vs. Time (color-coded by status)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.subplot2grid(shape = (3, 3), loc = (1, 2))

    invalid = df[df['status'] == 0]
    plt.plot(
        invalid["video_seconds"], 
        invalid["proj_gaze_mag_left"], 
        label= "Eyes not found", 
        alpha=0.8,
    )

    semivalid = df[df['status'] == 1]
    plt.plot(
        semivalid["video_seconds"], 
        semivalid["proj_gaze_mag_left"], 
        label= "Not yet calibrated", 
        alpha=0.8,
    )

    valid = df[df['status'] == 2]
    plt.plot(
        valid["video_seconds"], 
        valid["proj_gaze_mag_left"], 
        label= "Tracking online", 
        alpha=0.8,
    )

    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("2D Gaze Magnitude (normalized)")
    plt.title("Gaze Magnitude vs. Time (color-coded by status)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    


if __name__ == "__main__":
    parser = ap.ArgumentParser(description = "visualize recorded eye tracking")
    parser.add_argument('filenames', nargs='+', help='CSV files to process')
    args = parser.parse_args()
    for f in args.filenames:
        visualize(f)

