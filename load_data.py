"""
Author: Jesse Torres
Description: This module implements functions for loading
raw or adjusted data, adjusting data, splitting data into
training and test sets, and finding correlations and the
best lags between the proton and electron flux data sets.
"""

from scipy.stats import pearsonr
import numpy as np


def load_series(filename):
    """
    Loads a time series into a list.
    :param filename: The name of the file from which values will be read
    :return: A list of the values in the file
    """

    values = []
    file = open(filename, "r")
    line = file.readline()
    while line:
        line = line.split()
        values.append(float(line[0]))
        line = file.readline()

    file.close()
    return values


def load_2d(filename):
    """
    Load data to a two-dimensional array.
    :param filename: The name of the file to read from
    :return: A 2d array containing the data from the file
    """
    file = open(filename)
    data = []
    line = file.readline()
    while line:
        line = line.split()
        line = [float(value) for value in line]
        data.append(line)
        line = file.readline()
    file.close()
    return np.array(data)


def write_file(filename, data):
    """
    Write data to a file.
    :param filename: The name of the file to output to
    :param data: The list or array of data to write
    :return: Nothing
    """
    file = open(filename, "w")
    for value in data:
        file.write(f"{value}\n")
    file.close()


def write_2d(filename, data):
    """
    Write two-dimensional data to a file.
    :param filename: The name of the file to output to
    :param data: The list of lists or arrays to write
    :return: Nothing
    """
    file = open(filename, "w")
    for arr in data:
        for value in arr:
            file.write(f"{value} ")
        file.write("\n")
    file.close()


def load_so_data():
    """
    Load data from SO_derivative1.dat
    :return: The electron, proton, derivative of electron, and derivative of proton series, each as a list
    """

    # Open file and skip first four lines
    file = open("../../Data/SO_expave15m.dat", "r")  # Change to SO_derivative1.dat for time constant of 15 min
    for i in range(7):  # Change to 4 if using SO_derivative1.dat
        file.readline()

    # Time series to be read
    dates = []
    electron = []
    electron_high = []
    proton = []
    d_electron = []
    d_electron_err = []
    d_electron_high = []
    d_electron_high_err = []
    d_proton = []
    d_proton_err = []

    # Then, for each line, get the data according to the column numbers described in the file
    line = file.readline()
    while line:
        line = line.split()
        dates.append(line[0])
        electron.append(float(line[3]))  # Use line[3] for observed intensity, line[5] for fitted
        electron_high.append(float(line[9]))
        proton.append(float(line[15]))   # Use line[9] for observed intensity, line[11] for fitted
        d_electron.append(float(line[7]))
        d_electron_err.append(float(line[8]))
        d_electron_high.append(float(line[13]))
        d_electron_high_err.append(float(line[14]))
        d_proton.append(float(line[19]))
        d_proton_err.append(float(line[20]))
        line = file.readline()

    return dates, electron, electron_high, proton, d_electron, d_electron_err,\
        d_electron_high, d_electron_high_err, d_proton, d_proton_err


def load_so_events():
    """
    Load onset and peak indices of events.
    :return: A list of pairs, each of which is the onset and peak of an event
    """
    file = open("event_indices.txt", "r")
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    events = []
    for line in lines:
        line = line.split()
        onset = int(line[0])
        peak = int(line[2])
        events.append((onset, peak))
    return events


def load_so_events_test(n_train, n_test):
    """
    Load onset and peak indices of events, and shift indices for test set.
    :param n_train: The number of training instances
    :param n_test: The number of testing instances
    :return: A list of pairs, each of which is the onset and peak of an event
    """
    file = open("event_indices.txt", "r")
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    events = []
    for line in lines:
        line = line.split()
        onset = int(line[0])
        peak = int(line[2])
        if n_train < onset < n_train + n_test:
            events.append((onset - n_train - 24, peak - n_train - 24))
    return events


def adjust_series(time_series, mev):
    """
    Use counts to adjust zero/low-intesity values.
    :param time_series: A list of values containing a time series
    :param mev: 16.4 or 33
    :return: The adjusted time series along with nval which contains window size information
    """
    time_series_new = []
    nval = []
    for t in range(len(time_series)):

        # Initial f and C values
        intensity = time_series[t]
        seconds = 300
        if mev == 16.4:
            geometric_factor = 5.14
            mev_window = 17.2
        else:
            geometric_factor = 4.77
            mev_window = 16
        c = intensity * seconds * geometric_factor * mev_window

        # Find time window
        i = 0
        while c < 9 and t - 2 ** i >= 0:
            i += 1
            intensity = sum(time_series[max(0, t - 2 ** i): t])
            c = intensity * 2 ** i * seconds * geometric_factor * mev_window
        time_series_new.append(intensity)
        nval.append(2 ** i)

    return np.array(time_series_new), np.array(nval)


def find_correlations(x, y):
    """
    Calculate the correlation between two time series at different time lags.
    :param x: The first sequence, electron flux data
    :param y: The second sequence, proton flux data
    :return: A list of correlations in which each index is a time lag and contains
             the corresponding correlation; since time lag 0 is not considered, index
             0 contains the value 0
    """

    correlations = [0] * 37

    # Test time lags from 5 minutes to 3 hours in 5 minute increments
    for i in range(1, 37):

        # To lag the input, select all except for the last i * 5min
        # The output must have an input for all timesteps, so ignore the first i * 5min
        x_lagged = x[:len(x) - i]
        y_adjusted = y[i:]

        # Find correlation of current lag and add to list of correlations
        correlation, _ = pearsonr(x_lagged, y_adjusted)
        correlations[i] = correlation

    return correlations


def find_best_lags(correlations):
    """
    Find the indices of the top 6 correlations.
    :param correlations: A list of correlations for each lag
    :return: a list of the 6 indices with the highest correlations
    """

    # For first six, just place in dictionary
    # For rest, find if current lag has higher correlation than minimum in dictionary,
    # and if so, then replace minimum with current lag/correlation
    best_lags = {}
    for i, correlation in enumerate(correlations[1:]):
        if i + 1 > 6:
            min_correlation = min(best_lags.values())
            if correlation > min_correlation:
                for k in best_lags.keys():
                    if best_lags[k] == min_correlation:
                        best_lags.pop(k)
                        best_lags[i + 1] = correlation
                        break
        else:
            best_lags[i + 1] = correlation
    return best_lags.keys()


def pair_input_output(electron, proton, electron_deriv, proton_deriv,
                      use_proton_input, proton_lag, electron_lags):
    """
    Pair inputs to outputs such that the inputs are lagged behind the outputs.
    :param electron: A list of electron flux values
    :param proton: A list of proton flux values (cannot be None, unlike others)
    :param electron_deriv: A list of electron derivative values
    :param proton_deriv: A list of proton derivative values
    :param use_proton_input: A boolean indicating whether to use lagged proton flux
                             as a feature
    :param proton_lag: If using proton input, the closest input to the predicted value
                       at time t, in 5-minute increments (i.e. 2 = 10m, 6 = 30m, 12 = 1h)
    :param electron_lags: The lags between the electron inputs and predicted value at time t
    :return: The inputs and outputs, as well as a list of features for feature importance
    """

    # Start at t = 29 since proton lag is up to 2 hours (24 * 5), and derivatives require
    # the 5 previous values (therefore t - 29 must be accessible)
    x = []
    y = []
    features = []
    for t in range(29, len(proton)):

        # Start with output, which is a single value at time t
        y.append(proton[t])

        # Alternatively, try to predict difference between future and current values
        # y.append(proton[t] - proton[t - proton_lag])

        # Now get input for time t
        x_t = []
        if electron:
            for lag in electron_lags:
                x_t.append(electron[t - lag])
                if t == 29:
                    features.append(f"electron_t-{lag}")

        # Since proton is required to be present, it will be used as an input feature if
        # neither electron nor proton_deriv are present, even if use_proton_input is false
        if use_proton_input or not (electron or proton_deriv):
            for lag in range(proton_lag, 25):
                x_t.append(proton[t - lag])
                if t == 29:
                    features.append(f"proton_t-{lag}")

        if electron_deriv:
            for lag in electron_lags:
                x_t.append(electron_deriv[t - lag])
                if t == 29:
                    features.append(f"electron_deriv_t-{lag}")

        if proton_deriv:
            for lag in range(proton_lag, 25):
                x_t.append(proton_deriv[t - lag])
                if t == 29:
                    features.append(f"proton_deriv_t-{lag}")

        # Add x_t to list of all inputs
        x.append(x_t)

    return x, y, features


def train_test_split(x, y, train_ratio):
    """
    Split inputs and outputs into training and test sets.
    :param x: The inputs to the neural network
    :param y: The outputs corresponding to the given inputs
    :param train_ratio: The percentage of data to use for training
    :return: The training inputs and targets, and the test inputs and targets
    """
    train = x[:int(train_ratio * len(x))]
    targets_train = y[:int(train_ratio * len(y))]
    test = x[int(train_ratio * len(x)):]
    targets_test = y[int(train_ratio * len(y)):]
    return np.array(train), np.array(targets_train), np.array(test), np.array(targets_test)
