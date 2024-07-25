from keras.callbacks import EarlyStopping
from keras.layers import Dense, GRU
from keras.models import clone_model, load_model, Sequential
# from keras.utils import plot_model
from load_data import *
from stats import mae, x_axis_error, lag_ln10
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def slice_dictionary_keys(dictionary, start, end):
    """
    Slice a dictionary by keys rather than indices.
    :param dictionary: A dictionary object
    :param start: The starting key
    :param end: The ending key
    :return: The dictionary keys and values between the start and end keys
    """
    times = list(dictionary.keys())
    keys = times[times.index(start): times.index(end) + 1]
    return {key: dictionary[key] for key in keys}


def pair_input_output(data, use_phase_inputs, prediction_time):
    """
    Pair inputs and outputs, and split into training and testing sets.
    :param data: A DataFrame object
    :param use_phase_inputs: Whether or not to use phase inputs
    :param prediction_time: The number of timesteps to predict ahead
    :return: Training and test inputs and targets
    """

    # Select features
    time = list(data['time'].values)
    electron = data['electron'].values
    electron_high = data['electron_high'].values
    proton = data['proton'].values
    phases = np.array([data['background'].values, data['rising'].values, data['falling'].values]).T

    # Pair inputs and outputs
    x = []
    y = []
    for t in range(24, len(data) - prediction_time):

        # Get current instance (past 2 hours plus current)
        x_curr = [electron[t - 24: t + 1], electron_high[t - 24: t + 1], proton[t - 24: t + 1]]

        # Adjust phase inputs by checking if there was a change in phase before the past 30 minutes
        # If so, then starting from last index, go backwards and replace up to t-6 with the value of t-6
        if use_phase_inputs:
            phases_t = np.array(phases[t - 24: t + 1])
            i = 24  # 24 due to size of temporary phase array (instead of directly modifying entire phase data)
            while (phases_t[i] != phases_t[i - 6]).any():
                phases_t[i] = phases_t[i - 6]
                i -= 1

            # Add columns to current instance
            for j in range(phases_t.shape[1]):
                x_curr.append(phases_t[:, j])

        # Add to list of all instances
        x.append(x_curr)

        # Store output; include timestamps
        y.append([time[t + prediction_time], proton[t + prediction_time]])

    # Split train/test
    # Note that training targets do not need times, but test targets do for synchronization
    train = x[:int(0.8 * len(x))]
    targets_train = y[:int(0.8 * len(y))]
    test = x[int(0.8 * len(x)):]
    targets_test = y[int(0.8 * len(y)):]
    targets_test = {target[0]: target[1] for target in targets_test}
    return np.array(train), np.array(targets_train)[:, 1].astype(np.float64), np.array(test), targets_test


def estimate_thresholds(electron, electron_high, proton, prediction_time):
    """
    Find and output estimated thresholds for medium and high models; user can fine-tune these values.
    :param electron: The electron time series, as an array
    :param electron_high: The high-intensity electron time series, as an array
    :param proton: The proton time series, as an array
    :param prediction_time: The number of timesteps to predict ahead
    :return: Nothing
    """

    # Constants and variables to track during the following loop
    n = len(electron)
    ln10 = np.log(10)
    ln1 = np.log(1)
    count_high = 0
    count_medium = 0
    count_low = 0
    electron_threshold_high = max(electron)  # Start at max for each feature since we are trying to find min of max
    electron_threshold_medium = max(electron)
    proton_threshold_high = max(proton)
    proton_threshold_medium = max(proton)
    electron_high_threshold_high = max(electron_high)
    electron_high_threshold_medium = max(electron_high)

    # Find x threshold for medium and high models
    for t in range(24, n - prediction_time):

        # Event
        if proton[t + prediction_time] >= ln10:

            # Electron high threshold
            max_electron = max(electron[t - 24: t + 1])
            if max_electron < electron_threshold_high:
                electron_threshold_high = max_electron

            # High energy electron high threshold
            max_electron_high = max(electron_high[t - 24: t + 1])
            if max_electron_high < electron_high_threshold_high:
                electron_high_threshold_high = max_electron_high

            # Proton high threshold
            max_proton = max(proton[t - 24: t + 1])
            if max_proton < proton_threshold_high:
                proton_threshold_high = max_proton

            # Update count
            count_high += 1

        # In-between
        elif proton[t + prediction_time] >= ln1:

            # Electron medium threshold
            max_electron = max(electron[t - 24: t + 1])
            if max_electron < electron_threshold_medium:
                electron_threshold_medium = max_electron

            # High energy electron medium threshold
            max_electron_high = max(electron_high[t - 24: t + 1])
            if max_electron_high < electron_high_threshold_medium:
                electron_high_threshold_medium = max_electron_high

            # Proton medium threshold
            max_proton = max(proton[t - 24: t + 1])
            if max_proton < proton_threshold_medium:
                proton_threshold_medium = max_proton

            # Update count
            count_medium += 1

        # Background
        else:
            count_low += 1

    print(f"Calculated electron medium threshold = {electron_threshold_medium}, "
          f"calculated electron high threshold = {electron_threshold_high}")
    print(f"Calculated high energy electron medium threshold = {electron_high_threshold_medium}, "
          f"calculated high energy electron high threshold = {electron_high_threshold_high}")
    print(f"Calculated proton medium threshold = {proton_threshold_medium}, "
          f"calculated proton high threshold = {proton_threshold_high}")


def unpack_thresholds(thresholds):
    """
    Create variables for each threshold.
    :param thresholds: A dictionary containing a pair of thresholds for each feature
    :return: Each value in the dictionary, as an individual variable for each
    """
    medium_th_electron, high_th_electron = thresholds['electron']
    medium_th_electron_high, high_th_electron_high = thresholds['electron_high']
    medium_th_proton, high_th_proton = thresholds['proton']
    return medium_th_electron, high_th_electron, medium_th_electron_high, \
        high_th_electron_high, medium_th_proton, high_th_proton


def split_train(train, targets_train, thresholds):

    medium_th_electron, high_th_electron, medium_th_electron_high, high_th_electron_high,\
        medium_th_proton, high_th_proton = unpack_thresholds(thresholds)

    train_high = []
    targets_train_high = []
    train_medium = []
    targets_train_medium = []
    train_low = []
    targets_train_low = []
    count_high = 0
    count_medium = 0
    count_low = 0

    # For each instance, check if max value of past 5 hours exceeds threshold for any feature
    for i in range(len(train)):

        # First, find max of each feature
        max_electron = max(train[i][0])
        max_electron_high = max(train[i][1])
        max_proton = max(train[i][2])

        # Then check against thresholds, from highest to lowest
        if max_electron >= high_th_electron or max_electron_high >= high_th_electron_high or \
                max_proton >= high_th_proton:
            train_high.append(train[i])
            targets_train_high.append(targets_train[i])
            count_high += 1

        # Check against medium thresholds
        elif max_electron >= medium_th_electron or max_electron_high >= medium_th_electron_high or \
                max_proton >= medium_th_proton:
            train_medium.append(train[i])
            targets_train_medium.append(targets_train[i])
            count_medium += 1

        # Otherwise, use low model
        else:
            train_low.append(train[i])
            targets_train_low.append(targets_train[i])
            count_low += 1

    # print(f"Training set: count_low = {count_low}, count_medium = {count_medium}, count_high = {count_high}")
    return np.array(train_low), np.array(targets_train_low), np.array(train_medium), np.array(targets_train_medium),\
        np.array(train_high), np.array(targets_train_high)


def train_model(train_low, targets_train_low, train_medium, targets_train_medium,
                train_high, targets_train_high, algorithm):

    # Reshape training sets depending on algorithm
    if algorithm == 'regular':
        train_low = np.array([instance.flatten() for instance in train_low])
        train_medium = np.array([instance.flatten() for instance in train_medium])
        train_high = np.array([instance.flatten() for instance in train_high])
    else:
        train_low = np.array([instance.T for instance in train_low])
        train_medium = np.array([instance.T for instance in train_medium])
        train_high = np.array([instance.T for instance in train_high])

    # Create low model
    low_model = Sequential()
    if algorithm == 'regular':
        low_model.add(Dense(30, input_shape=train_low.shape[1:], activation='sigmoid'))
    else:
        low_model.add(GRU(30, input_shape=train_low.shape[1:], activation='sigmoid', return_sequences=False))
    low_model.add(Dense(1))

    # Medium and high models are identical to low model; initial weights will be different after each is compiled
    medium_model = clone_model(low_model)
    high_model = clone_model(low_model)
    low_model.compile(loss='mse', optimizer='adam')
    medium_model.compile(loss='mse', optimizer='adam')
    high_model.compile(loss='mse', optimizer='adam')

    # Train and return models
    low_model.fit(train_low, targets_train_low, epochs=1000, verbose=1,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    medium_model.fit(train_medium, targets_train_medium, epochs=1000, verbose=1,
                     callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    high_model.fit(train_high, targets_train_high, epochs=1000, verbose=1,
                   callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    return low_model, medium_model, high_model


def predict(test, low_model, medium_model, high_model, algorithm, thresholds):

    medium_th_electron, high_th_electron, medium_th_electron_high, high_th_electron_high,\
        medium_th_proton, high_th_proton = unpack_thresholds(thresholds)

    model_series = []
    count_high = 0
    count_medium = 0
    count_low = 0

    # For each instance, check if max value of past 5 hours exceeds threshold for any feature
    for i in range(len(test)):

        # First, find max of each feature
        max_electron = max(test[i][0])
        max_electron_high = max(test[i][1])
        max_proton = max(test[i][2])

        # Then check against thresholds, from highest to lowest
        if max_electron >= high_th_electron or max_electron_high >= high_th_electron_high or \
                max_proton >= high_th_proton:
            # predictions.append(high_model.predict(instance)[0][0])
            model_series.append(-6)
            count_high += 1

        # Check against medium thresholds
        elif max_electron >= medium_th_electron or max_electron_high >= medium_th_electron_high or \
                max_proton >= medium_th_proton:
            # predictions.append(medium_model.predict(instance)[0][0])
            model_series.append(-7)
            count_medium += 1

        # Otherwise, use low model
        else:
            # predictions.append(low_model.predict(instance)[0][0])
            model_series.append(-8)
            count_low += 1

    # Reshape test set depending on algorithm
    if algorithm == 'regular':
        test = np.array([instance.flatten() for instance in test])
    else:
        test = np.array([instance.T for instance in test])

    # Using indices, predict from each of the three models
    model_series = np.array(model_series)
    predictions = np.zeros(len(test))  # Initialize list and populate with each model's predictions
    ind_low = np.where(model_series == -8)
    predictions[ind_low] = low_model.predict(test[ind_low]).flatten()
    ind_medium = np.where(model_series == -7)
    predictions[ind_medium] = medium_model.predict(test[ind_medium]).flatten()
    ind_high = np.where(model_series == -6)
    predictions[ind_high] = high_model.predict(test[ind_high]).flatten()

    # print(f"Test set: count_low = {count_low}, count_medium = {count_medium}, count_high = {count_high}")
    return np.array(predictions), model_series


def evaluate(targets_test, predictions, model_series, event_times, data, path, display):

    # First, get electron, high energy electron, and xray from data with timestamps for plots
    times = list(data["time"].values)
    electron = data["electron"].values
    electron_high = data["electron_high"].values

    # Convert to dictionaries
    electron_dict = {times[i]: electron[i] for i in range(len(electron) - len(targets_test), len(electron))}
    electron_high_dict = {times[i]: electron_high[i] for i in range(len(electron_high) - len(targets_test),
                                                                    len(electron_high))}

    # Verify that times synchronize with test targets
    if targets_test.keys() != electron_dict.keys() != electron_high_dict.keys():
        print("Timestamps are not synchronized, exiting program.")
        exit(0)

    # Begin evaluating events
    maes = []
    o2p_lags = []
    o2t_lags = []
    ln10_lags = []
    for i, event in enumerate(event_times):

        # Get event times
        onset = event[0]
        bg_before_onset = times[times.index(onset) - 36]
        threshold = event[1]
        peak = event[2]

        # Get relevant portions of each time series for plotting
        times_o2p = times[times.index(bg_before_onset): times.index(peak) + 1]
        targets_o2p = slice_dictionary_keys(targets_test, bg_before_onset, peak)
        predictions_o2p = slice_dictionary_keys(predictions, bg_before_onset, peak)
        electron_o2p = slice_dictionary_keys(electron_dict, bg_before_onset, peak)
        electron_high_o2p = slice_dictionary_keys(electron_high_dict, bg_before_onset, peak)
        model_series_o2p = slice_dictionary_keys(model_series, bg_before_onset, peak)

        # Plot event
        fig, ax = plt.subplots()
        ax.set_ylabel('ln(Flux (/cc/s/sr))')
        ax.plot(list(targets_o2p.values()), '-b', label='Actual proton')
        ax.plot(list(predictions_o2p.values()), '-r', label='Predicted proton')
        ax.plot(list(electron_o2p.values()), '-m', label='Electron')
        ax.plot(list(electron_high_o2p.values()), '-y', label='High-energy electron')
        ax.plot(list(model_series_o2p.values()), '-g', label='Model')
        ax.plot([np.log(10)] * len(targets_o2p.values()), '--k')

        # Set x-axis to timestamps
        diff = int(ax.get_xticks()[1] - ax.get_xticks()[0])
        time_labels = [time[time.index('T') + 1: time.index('.')] for time in times_o2p[::diff]]
        if len(time_labels) == len(ax.get_xticks()) - 2:
            ax.set_xticks(ax.get_xticks()[1:-1])
        else:
            ax.set_xticks(ax.get_xticks()[1:-2])
        ax.set_xticklabels(time_labels, rotation='vertical')

        # Caption with date
        date_start = times_o2p[0][:times_o2p[0].index('T')]
        date_end = times_o2p[-1][:times_o2p[-1].index('T')]
        if date_start != date_end:
            ax.set_xlabel(f'Event from {date_start} to {date_end}')
        else:
            ax.set_xlabel(f'Event on {date_start}')

        fig.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
        fig.tight_layout()
        if path:
            plt.savefig(f'{path}/event{i + 1}.png')
        if display:
            plt.show()
        plt.close()

        # Now remove the 3 hours of background since they are not evaluated
        targets_o2p = slice_dictionary_keys(targets_test, onset, peak)
        predictions_o2p = slice_dictionary_keys(predictions, onset, peak)
        targets_o2t = slice_dictionary_keys(targets_test, onset, threshold)
        predictions_o2t = slice_dictionary_keys(predictions, onset, threshold)

        # Also convert from dictionary to numpy array; timestamps are no longer needed
        targets_o2p = np.array(list(targets_o2p.values()))
        predictions_o2p = np.array(list(predictions_o2p.values()))
        targets_o2t = np.array(list(targets_o2t.values()))
        predictions_o2t = np.array(list(predictions_o2t.values()))

        # Calculate stats
        maes.append(mae(targets_o2p, predictions_o2p, False))
        o2p_lags.append(x_axis_error(targets_o2p, predictions_o2p, False))
        o2t_lags.append(x_axis_error(targets_o2t, predictions_o2t, False))
        ln10_lags.append(lag_ln10(targets_o2p, predictions_o2p))

    # Output metrics per event
    if path:
        outfile = open(f'{path}/results.txt', 'w')
        for i in range(len(event_times)):
            outfile.write(f"Event {i + 1}\nMAE = {maes[i]: 0.3f}\nO2P lag = {o2p_lags[i]: 0.3f}\n"
                          f"O2T lag = {o2t_lags[i]: 0.3f}\nln10 lag = {ln10_lags[i]: 0.3f}\n\n")
        outfile.write(f"Average MAE = {np.average(maes): 0.3f}\n")
        outfile.write(f"Average O2P lag = {np.average(o2p_lags): 0.3f}\n")
        outfile.write(f"Average O2T lag = {np.average(o2t_lags): 0.3f}\n")
        outfile.write(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}\n")
        outfile.close()

    # Output average metrics to standard output
    print(f"Average MAE = {np.average(maes): 0.3f}")
    print(f"Average O2P lag = {np.average(o2p_lags): 0.3f}")
    print(f"Average O2T lag = {np.average(o2t_lags): 0.3f}")
    print(f"Average ln10 lag = {np.average(ln10_lags): 0.3f}")


def main():

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--prediction_time', type=int, required=False,
                        default=6, help="Number of timesteps ahead to predict")
    parser.add_argument('-a', '--algorithm', type=str, required=False, default='regular',
                        choices=['regular', 'rnn'], help="Algorithm to be used for both model selection and "
                                                         "intensity models (regular or RNN)")
    parser.add_argument('-ph', '--phase_inputs', action='store_true', required=False,
                        default=False, help="Whether or not to use phase inputs")
    parser.add_argument('-lm', '--load_model', action='store_true', required=False,
                        default=False, help="Whether or not to load trained model")
    parser.add_argument('-lp', '--load_predictions', action='store_true', required=False,
                        default=False, help="Whether or not to load predictions")
    parser.add_argument('-p', '--path', type=str, required=False,
                        default=None, help="Directory to load and store files")
    parser.add_argument('-d', '--display', action='store_true', required=False, default=False,
                        help="Whether or not to display event plots")
    parser.add_argument('-s', '--seed', type=int, required=False, help='Seed for random number generator')

    # Parse arguments
    args = parser.parse_args()
    prediction_time = args.prediction_time
    algorithm = args.algorithm
    use_phase_inputs = args.phase_inputs
    load_models = args.load_model
    load_predictions = args.load_predictions
    path = args.path
    display = args.display
    seed = args.seed

    # Pair inputs and outputs, and split into train/test
    # The feature in the command line argument is used for switching, but xray integral
    # is always used for intensity (unless no xray is used)
    data = pd.read_csv('../Data/data.csv')
    train, targets_train, test, targets_test = pair_input_output(data, use_phase_inputs, prediction_time)

    # Initial estimate of thresholds; use these results for further fine tuning in lines 461-464
    # estimate_thresholds(data['electron'].values, data['electron_high'].values, data['proton'].values, prediction_time)

    # Set thresholds (different based on t+6/12)
    if prediction_time == 6:
        thresholds = {'electron': (2.2, 6.2), 'electron_high': (3, 5), 'proton': (-3, 4.8)}
    elif prediction_time == 12:
        thresholds = {'electron': (2, 5.8), 'electron_high': (2.3, 5.5), 'proton': (-5, 1.5)}
    else:
        print(f"Need to set thresholds for t + {prediction_time}, exiting program.")
        exit(0)

    # Split into ranges based on thresholds
    train_low, targets_train_low, train_medium, targets_train_medium, train_high, targets_train_high = \
        split_train(train, targets_train, thresholds)

    # Either load predictions, load model and get predictions, or train model and get predictions
    if load_predictions:
        if path:
            predictions = load_series(f"{path}/predictions.txt")
        else:
            print("No path to load predictions from, exiting program.")
            exit(0)
    else:
        if load_models:
            if path:
                low_model = load_model(f"{path}/low_model")
                medium_model = load_model(f"{path}/medium_model")
                high_model = load_model(f"{path}/high_model")
            else:
                print("No path to load model from, exiting program.")
                exit(0)
        else:
            low_model, medium_model, high_model = train_model(train_low, targets_train_low, train_medium,
                                                              targets_train_medium, train_high, targets_train_high,
                                                              algorithm)
            if path:
                low_model.save(f"{path}/low_model")
                medium_model.save(f"{path}/medium_model")
                high_model.save(f"{path}/high_model")

        # Use predict function for multiple models/thresholds
        predictions, model_series = predict(test, low_model, medium_model, high_model, algorithm, thresholds)
        if path:
            write_file(f"{path}/predictions.txt", predictions)

    # Load events for evaluation
    event_file = open('../Data/event_timestamps.txt', 'r')
    lines = event_file.readlines()
    event_times = [line.split() for line in lines]
    event_file.close()

    # Select only events which are in the test set
    event_times_test = []
    first_test_event_time = list(targets_test.keys())[0]
    for event in event_times:
        if event[0] >= first_test_event_time:
            event_times_test.append(event)

    # Evaluate predictions
    target_times = list(targets_test.keys())
    predictions = {target_times[i]: predictions[i] for i in range(len(predictions))}
    model_series = {target_times[i]: model_series[i] for i in range(len(model_series))}
    evaluate(targets_test, predictions, model_series, event_times_test, data, path, display)


if __name__ == "__main__":
    main()
