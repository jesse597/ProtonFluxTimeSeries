import numpy as np
import sys

# Path should contain directories each containing the results of a single run
path = sys.argv[1]
maes = []
o2p_lags = []
o2t_lags = []
ln10_lags = []

for i in range(5):
    if path.__contains__("M3_ML"):
        results_file = open(f"{path}/run{i + 1}/results_intensity.txt", "r")
    else:
        results_file = open(f"{path}/run{i + 1}/results.txt", "r")
    lines = results_file.readlines()[108:]  # 108 = 18 * 6 = number of events * lines per event
    mae_line, o2p_line, o2t_line, ln10_line = lines[:4]
    maes.append(float(mae_line[mae_line.index('=') + 1:]))
    o2p_lags.append(float(o2p_line[o2p_line.index('=') + 1:]))
    o2t_lags.append(float(o2t_line[o2t_line.index('=') + 1:]))
    ln10_lags.append(float(ln10_line[ln10_line.index('=') + 1:]))
    results_file.close()

outfile = open(f"{path}/avg_results.txt", "w")
outfile.write(f"Average MAE = {np.mean(maes):.3f} (+/- {np.std(maes):.3f})\n")
outfile.write(f"Average O2P lag = {np.mean(o2p_lags):.3f} (+/- {np.std(o2p_lags):.3f})\n")
outfile.write(f"Average O2T lag = {np.mean(o2t_lags):.3f} (+/- {np.std(o2t_lags):.3f})\n")
outfile.write(f"Average ln10 lag = {np.mean(ln10_lags):.3f} (+/- {np.std(ln10_lags):.3f})\n")
outfile.close()
