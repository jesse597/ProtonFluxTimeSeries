"""
Author: Jesse Torres
Description: This program matches timestamps from p10f10event.dat to timestamps in SO_expave15m.dat
in order to find the indices of events. From this, the onset-to-peak duration, peak intensity, and lag between
electron and proton are analyzed.
"""
from load_data import *
import matplotlib.pyplot as plt
import numpy as np
import sys


def load_indices():
    """
    Get indices for each event's onset, threshold, peak, and end.
    :return: Four lists, one for each part of the event
    """

    # Indices to keep track of
    onset_indices = []
    threshold_indices = []
    peak_indices = []
    end_indices = []

    file = open("event_indices.txt", "r")
    lines = file.readlines()
    for line in lines:
        line = line.split()
        onset_indices.append(int(line[0]))
        threshold_indices.append(int(line[1]))
        peak_indices.append(int(line[2]))
        end_indices.append(int(line[3]))

    return onset_indices, threshold_indices, peak_indices, end_indices


def find_intensities(electron, electron_high, proton, onset_indices, threshold_indices, peak_indices):

    # Stats to keep track of
    onset_electron_intensities = []
    onset_high_electron_intensities = []
    onset_proton_intensities = []
    threshold_electron_intensities = []
    threshold_high_electron_intensities = []
    threshold_proton_intensities = []
    peak_electron_intensities = []
    peak_high_electron_intensities = []
    peak_proton_intensities = []

    # Loop through each event to find intensities
    for i in range(len(onset_indices)):
        onset_electron_intensities.append(electron[onset_indices[i]])
        onset_high_electron_intensities.append(electron_high[onset_indices[i]])
        onset_proton_intensities.append(proton[onset_indices[i]])
        threshold_electron_intensities.append(electron[threshold_indices[i]])
        threshold_high_electron_intensities.append(electron_high[threshold_indices[i]])
        threshold_proton_intensities.append(proton[threshold_indices[i]])
        peak_electron_intensities.append(electron[peak_indices[i]])
        peak_high_electron_intensities.append(electron_high[peak_indices[i]])
        peak_proton_intensities.append(proton[peak_indices[i]])

    return onset_electron_intensities, onset_high_electron_intensities, onset_proton_intensities, \
           threshold_electron_intensities, threshold_high_electron_intensities, threshold_proton_intensities, \
           peak_electron_intensities, peak_high_electron_intensities, peak_proton_intensities


def find_durations(onset_indices, peak_indices):

    durations = []
    for i in range(len(onset_indices)):
        durations.append((peak_indices[i] - onset_indices[i]) / 12)
    return durations


def find_correlations(electron, electron_high, proton, onset_indices, peak_indices, n_lags):

    corrs = []
    corrs_high = []
    corrs_transpose = [[] for i in range(n_lags)]
    corrs_high_transpose = [[] for i in range(n_lags)]
    from tqdm import tqdm
    for i in tqdm(range(len(onset_indices))):

        onset = onset_indices[i]
        peak = peak_indices[i]

        # Find lag with highest Pearson correlation (up to 3 hours)
        corrs_event = []
        corrs_high_event = []
        for lag in range(min(n_lags, (peak - onset) - 1)):

            # Shift electrons forward to lag, and adjust protons so that lengths are the same
            electron_lagged = electron[:len(electron) - lag]
            electron_high_lagged = electron_high[:len(electron_high) - lag]
            proton_adjusted = proton[lag:]
            corrs_event.append(pearsonr(electron_lagged[onset:peak], proton_adjusted[onset:peak])[0])
            corrs_high_event.append(pearsonr(electron_high_lagged[onset:peak], proton_adjusted[onset:peak])[0])

        # Add to list of lists of correlations for each event
        corrs.append(np.array(corrs_event))
        corrs_high.append(np.array(corrs_high_event))
        for j in range(len(corrs_event)):
            corrs_transpose[j].append(corrs_event[j])
            corrs_high_transpose[j].append(corrs_high_event[j])

    return np.array(corrs), np.array(corrs_high), np.array(corrs_transpose), np.array(corrs_high_transpose)


def main():

    # Get event indices and intensity data
    onset_indices, threshold_indices, peak_indices, end_indices = load_indices()
    _, electron, electron_high, proton, _, _, _, _, _, _ = load_so_data()

    # Get intensity data for events
    onset_electron_intensities, onset_high_electron_intensities, onset_proton_intensities, \
        threshold_electron_intensities, threshold_high_electron_intensities, threshold_proton_intensities, \
        peak_electron_intensities, peak_high_electron_intensities, peak_proton_intensities = \
        find_intensities(electron, electron_high, proton, onset_indices, threshold_indices, peak_indices)

    # Get duration of events from onset to peak
    durations_onset_peak = find_durations(onset_indices, peak_indices)

    # Use correlation to find lag between electron and proton for all events
    n_lags = int(sys.argv[1])
    corrs, corrs_high, corrs_transpose, corrs_high_transpose = \
        find_correlations(electron, electron_high, proton, onset_indices, peak_indices, n_lags)

    # Stats for all events
    lags = []
    lags_high = []
    max_corrs = []
    max_corrs_high = []

    for i in range(len(corrs)):

        # Full dataset lags and correlation
        lags.append(corrs[i].argmax())
        lags_high.append(corrs_high[i].argmax())
        max_corrs.append(max(corrs[i]))
        max_corrs_high.append(max(corrs_high[i]))

    # print("Max correlations")
    # for value in max_corrs:
    #     print(value)
    #
    # print("\nMax correlations high")
    # for value in max_corrs_high:
    #     print(value)

    # Scatter plot with each of best correlations (y) and their corresponding lags (x)
    plt.scatter(lags, max_corrs)
    plt.title('Lag vs. best correlation scatter plot')
    plt.xlabel('Lag (hours)')
    plt.ylabel('Max correlation of each event')
    plt.xticks(np.linspace(0, n_lags, n_lags // 12 + 1), np.arange(n_lags // 12 + 1))
    plt.show()
    plt.close()

    # Scatter plot with each of best correlations and corresponding lags for high energy electron
    plt.scatter(lags_high, max_corrs_high)
    plt.title('Lag vs. best correlation scatter plot (high energy electron)')
    plt.xlabel('Lag (hours)')
    plt.ylabel('Max correlation of each event')
    plt.xticks(np.linspace(0, n_lags, n_lags // 12 + 1), np.arange(n_lags // 12 + 1))
    plt.show()
    plt.close()

    # Plot distribution of onset-to-peak durations
    plt.hist([durations_onset_peak[:21], durations_onset_peak[21:]],
             label=["Train", "Test"])
    plt.title("Distribution of onset-to-peak durations")
    plt.xlabel("Onset-to-peak duration (hours)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.close()

    # Plot distribution of peak proton intensities
    plt.hist([peak_proton_intensities[:21], peak_proton_intensities[21:]],
             label=["Train", "Test"])
    plt.title("Distribution of peak proton intensities")
    plt.xlabel("Peak proton ln(Flux (/cc/s/sr))")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
