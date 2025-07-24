import pandas as pd
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt

def compare(args):
    filenames = args.filenames
    start_idx = args.start_index
    end_idx = args.end_index

    dfs = []
    len = 0
    for i, filename in enumerate(filenames):
        df = pd.read_csv(filename, delimiter = '\t').iloc[start_idx:end_idx]
        if i == 0:
            end_idx = df.__len__()
        dfs.append(df)
        len = i + 1

    fig, axes = plt.subplots(3, 5, figsize = (24, 12))

    ewriter = pd.ExcelWriter('compareexp.xlsx', engine = 'openpyxl')

    for i, filename_i in enumerate(args.filenames):
        for j, filename_j in enumerate(args.filenames):
            if j > i:
                # print(dfs[i][dfs[i].columns[2]])
                # print(dfs[j][dfs[j].columns[2]])
                comparison_df = dfs[i][dfs[i].columns[2]] == dfs[j][dfs[j].columns[2]]

                print('i : ' + str(i) + ', j: ' + str(j))
                print(comparison_df.value_counts(normalize = True) * 100)

                pd.DataFrame(comparison_df).to_csv('dump/' + str(i) + str(j) + '.csv', index = False)

    






    

if __name__ == "__main__":
    parser = ap.ArgumentParser(description = "compare classification files")
    parser.add_argument('filenames', nargs='+', help='CSV files to process')
    parser.add_argument('-s', '--start_index', default = 0, help = 'start index')
    parser.add_argument('-e', '--end_index', default = -1, help = 'end index')
    args = parser.parse_args()
    compare(args)

