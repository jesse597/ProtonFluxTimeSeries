from keras.callbacks import EarlyStopping
from keras.layers import Dense, GRU
from keras.models import load_model, Sequential
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


def pair_input_output(data, xray, prediction_time):
    """
    Pair inputs and outputs, and split into training and testing sets.
    :param data: A DataFrame object
    :param xray: An array containing values of a derived feature related to xray, or None
    :param prediction_time: The number of timesteps to predict ahead
    :return: Training and test inputs and targets
    """

    # Select features
    time = list(data['time'].values)
    electron = data['electron'].values
    electron_high = data['electron_high'].values
    proton = data['proton'].values

    # Pair inputs and outputs
    x = []
    y = []
    for t in range(60, len(data) - prediction_time):

        # Get current instance (past 5 hours plus current)
        x_curr = [electron[t - 60: t + 1], electron_high[t - 60: t + 1], proton[t - 60: t + 1]]
        if xray is not None:
            x_curr.append(xray[t - 60: t + 1])

        # Add to list of all instances
        x.append(x_curr)

        # Store output; include timestamps
        y.append([time[t + prediction_time], proton[t + prediction_time]])

    # Split train/test
    # Timestamp for splitting depends on prediction time, but both are close to each other
    # Timestamps are hardcoded so that they split at the same timestamp as the non-interpolated data
    if prediction_time == 6:
        time_split = '2000-12-17T03:40:00.000'
    elif prediction_time == 12:
        time_split = '2000-12-17T03:45:00.000'
    split_index = list(np.array(y)[:, 0]).index(time_split)

    # Note that training targets do not need times, but test targets do for synchronization
    train = x[:split_index]
    targets_train = y[:split_index]
    test = x[split_index:]
    targets_test = y[split_index:]
    targets_test = {target[0]: target[1] for target in targets_test}
    return np.array(train), np.array(targets_train)[:, 1].astype(np.float64), np.array(test), targets_test


def train_model(train, targets_train, algorithm):
    """
    Create and train the model.
    :param train: A numpy array in which each row is an instance
    :param targets_train: The targets for each training instance
    :param algorithm: 'regular' or 'rnn'
    :return: The trained model
    """
    model = Sequential()
    if algorithm == 'regular':
        model.add(Dense(30, input_shape=train.shape[1:], activation='relu'))
    else:
        model.add(GRU(30, input_shape=train.shape[1:], activation='sigmoid', return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train, targets_train, epochs=1000, verbose=1,
              callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    return model


def evaluate(targets_test, predictions, event_times, xray, data, path, display):

    # First, get electron, high energy electron, and xray from data with timestamps for plots
    times = list(data["time"].values)
    electron = data["electron"].values
    electron_high = data["electron_high"].values

    # Convert to dictionaries
    electron_dict = {times[i]: electron[i] for i in range(len(electron) - len(targets_test), len(electron))}
    electron_high_dict = {times[i]: electron_high[i] for i in range(len(electron_high) - len(targets_test),
                                                                    len(electron_high))}
    if xray is not None:
        xray_dict = {times[i]: xray[i] for i in range(len(xray) - len(targets_test), len(xray))}

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
        if xray is not None:
            xray_o2p = slice_dictionary_keys(xray_dict, bg_before_onset, peak)

        # Plot event
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('ln(Flux (/cc/s/sr))')
        ax1.plot(list(targets_o2p.values()), '-b', label='Actual proton')
        ax1.plot(list(predictions_o2p.values()), '-r', label='Predicted proton')
        ax1.plot(list(electron_o2p.values()), '-m', label='Electron')
        ax1.plot(list(electron_high_o2p.values()), '-y', label='High-energy electron')
        if xray is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Xray Flux')
            ax2.plot(list(xray_o2p.values()), '-k', label='X-ray')

        # Set x-axis to timestamps
        diff = int(ax1.get_xticks()[1] - ax1.get_xticks()[0])
        time_labels = [time[time.index('T') + 1: time.index('.')] for time in times_o2p[::diff]]
        if len(time_labels) == len(ax1.get_xticks()) - 2:
            ax1.set_xticks(ax1.get_xticks()[1:-1])
        else:
            ax1.set_xticks(ax1.get_xticks()[1:-2])
        ax1.set_xticklabels(time_labels, rotation='vertical')

        # Caption with date
        date_start = times_o2p[0][:times_o2p[0].index('T')]
        date_end = times_o2p[-1][:times_o2p[-1].index('T')]
        if date_start != date_end:
            ax1.set_xlabel(f'Event from {date_start} to {date_end}')
        else:
            ax1.set_xlabel(f'Event on {date_start}')

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
            outfile.write(f"Event {i + 1}\nMAE = {maes[i]: 0.4f}\nO2P lag = {o2p_lags[i]: 0.4f}\n"
                          f"O2T lag = {o2t_lags[i]: 0.4f}\nln10 lag = {ln10_lags[i]: 0.4f}\n\n")
        outfile.write(f"Average MAE = {np.average(maes): 0.4f}\n")
        outfile.write(f"Average O2P lag = {np.average(o2p_lags): 0.4f}\n")
        outfile.write(f"Average O2T lag = {np.average(o2t_lags): 0.4f}\n")
        outfile.write(f"Average ln10 lag = {np.average(ln10_lags): 0.4f}\n")
        outfile.close()

    # Output average metrics to standard output
    print(f"Average MAE = {np.average(maes): 0.4f}")
    print(f"Average O2P lag = {np.average(o2p_lags): 0.4f}")
    print(f"Average O2T lag = {np.average(o2t_lags): 0.4f}")
    print(f"Average ln10 lag = {np.average(ln10_lags): 0.4f}")


def main():

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--xray_feature', type=str, required=False,
                        help="Name of xray feature to use in addition to electron and proton"
                             "(must be name of a column from derived features table)")
    parser.add_argument('-t', '--prediction_time', type=int, required=False,
                        default=6, help="Number of timesteps ahead to predict")
    parser.add_argument('-a', '--algorithm', type=str, required=False, default='regular',
                        choices=['regular', 'rnn'], help="Algorithm to be used for intensity model (regular or RNN)")
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
    xray_feature = args.xray_feature
    prediction_time = args.prediction_time
    algorithm = args.algorithm
    load_models = args.load_model
    load_predictions = args.load_predictions
    path = args.path
    display = args.display
    seed = args.seed

    # Pair inputs and outputs, and split into train/test
    data_observed = pd.read_csv('../Data/data_interpolated_observed.csv')
    if xray_feature:
        data_derived = pd.read_csv('../Data/data_interpolated_derived.csv')
        xray = data_derived[xray_feature].values
    else:
        xray = None
    train, targets_train, test, targets_test = pair_input_output(data_observed, xray, prediction_time)

    # Reshape train/test depending on algorithm
    if algorithm == 'regular':
        train = np.array([instance.flatten() for instance in train])
        test = np.array([instance.flatten() for instance in test])
    else:
        train = np.array([instance.T for instance in train])
        test = np.array([instance.T for instance in test])

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
                model = load_model(f"{path}/model")
            else:
                print("No path to load model from, exiting program.")
                exit(0)
        else:
            model = train_model(train, targets_train, algorithm)
            if path:
                model.save(f"{path}/model")

        # Get and save predictions from model
        predictions = model.predict(test)
        predictions = predictions.flatten()
        if path:
            write_file(f"{path}/predictions.txt", predictions)

    # Load events for evaluation
    event_file = open('../Data/event_timestamps_interpolated.txt', 'r')
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
    evaluate(targets_test, predictions, event_times_test, xray, data_observed, path, display)


if __name__ == "__main__":
    main()
