from keras.callbacks import EarlyStopping
from keras.layers import Dense, GRU
from keras.models import clone_model, load_model, Sequential
# from keras.utils import plot_model
from load_data import *
from sklearn.metrics import confusion_matrix
from stats import mae, x_axis_error, lag_ln10
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Global constants for the three classes
BACKGROUND = 0
RISING = 1
FALLING = 2


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


def pair_input_output_switch(data, use_phase_inputs, prediction_time):
    """
    Pair inputs and outputs, and split into training and testing sets.
    :param data: A DataFrame object
    :param use_phase_inputs: Whether or not to use phase inputs
    :param prediction_time: The number of timesteps to predict ahead
    :return: Training and test inputs and targets
    """

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

        # Store output phase
        y.append(phases[t + prediction_time])

    # Split train/test
    # Targets do not need to keep timestamps in this case (as opposed to intensity models), since no plotting
    train = x[:int(0.8 * len(x))]
    targets_train = y[:int(0.8 * len(y))]
    test = x[int(0.8 * len(x)):]
    targets_test = y[int(0.8 * len(y)):]
    return np.array(train), np.array(targets_train), np.array(test), np.array(targets_test)


def find_sample_weights(targets_train):
    counts = np.bincount(targets_train.argmax(axis=1))  # Order is background, rising, falling
    ratios = [1, 3 * counts[0] / counts[1], counts[0] / counts[2]]  # Rising becomes 3x more important than background
    sample_weights = []
    for label in targets_train:
        sample_weights.append(ratios[label.argmax()])
    return np.array(sample_weights)


def train_phase_selection_model(train, targets_train, sample_weights, algorithm):
    """
    Train a classifier network on the given training set.
    :param train: A 2D numpy array containing the training instances
    :param targets_train: A 1D numpy array containing the training targets
    :param sample_weights: A 1D numpy array containing a weight for each training instance
    :param algorithm: regular or rnn
    :return: The trained model
    """

    # Create phase selection model
    model = Sequential()
    if algorithm == 'regular':
        model.add(Dense(30, input_shape=train.shape[1:], activation='sigmoid'))
    else:
        model.add(GRU(30, input_shape=train.shape[1:], activation='sigmoid', return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train and return models
    model.fit(train, targets_train, epochs=1000, sample_weight=sample_weights, verbose=1,
              callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    return model


def evaluate_phase_selection_model(targets_test, predictions, path=None):
    """
    Use a confusion matrix to evaluate the performance of the network.
    :param targets_test: A list of lists of length three, which are one-of-n representations of the three classes
    :param predictions: A list of the same shape as targets_test, output from the trained neural net
    :param path: A string containing the directory to output results to
    :return: Nothing
    """
    targets_test = np.array(targets_test)
    predictions = np.array(predictions)

    # Find column with highest value so that targets_test and predictions are lists of single values
    mtx = confusion_matrix(targets_test.argmax(axis=1), predictions.argmax(axis=1))
    print("Rows are actual, columns are predicted")
    print("Order is background, rising, falling")
    print(mtx)
    precision = mtx[1][1] / sum(mtx[:, 1])
    recall = mtx[1][1] / sum(mtx[1])
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Rising precision = {precision: 0.3f}")
    print(f"Rising recall = {recall: 0.3f}")
    print(f"Rising F1 = {f1: 0.3f}")

    # Output to file as well, if a path is provided
    if path:
        outfile = open(f"{path}/results_selection.txt", "w")
        outfile.write("Rows are actual, columns are predicted\n")
        outfile.write("Order is background, rising, falling\n")
        outfile.write(str(mtx))
        outfile.write(f"\nRising precision = {precision: 0.3f}\n")
        outfile.write(f"Rising recall = {recall: 0.3f}\n")
        outfile.write(f"Rising F1 = {f1: 0.3f}\n")
        outfile.close()


def pair_input_output_intensity(data, use_phase_inputs, prediction_time):
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

        # Get current instance (past 5 hours plus current)
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


def split_train_intensity(model_selector, train, targets_train):
    """
    Split the training set by predicted phase, with proton intensity as the new target.
    These will be used for training the three models.
    :param model_selector: The trained network for predicting the future phase
    :param train:
    :param targets_train:
    :return: The three training sets, along with their target values
    """

    # Training sets for intensity prediction
    train_background = []
    targets_train_background = []
    train_rising = []
    targets_train_rising = []
    train_falling = []
    targets_train_falling = []

    # Based on model predicted for time t, add instance at time t to corresponding training set
    phases = model_selector.predict(train).argmax(axis=1)
    for t in range(len(train)):
        if phases[t] == BACKGROUND:
            train_background.append(train[t])
            targets_train_background.append(targets_train[t])
        elif phases[t] == FALLING:
            train_falling.append(train[t])
            targets_train_falling.append(targets_train[t])
        else:
            train_rising.append(train[t])
            targets_train_rising.append(targets_train[t])

    return np.array(train_background), np.array(targets_train_background), \
        np.array(train_rising), np.array(targets_train_rising), \
        np.array(train_falling), np.array(targets_train_falling)


def train_intensity_models(train_background, targets_train_background, train_rising, targets_train_rising,
                           train_falling, targets_train_falling, algorithm):
    """
    Train the three intensity models for each phase.
    :param train_background: A 2D array of the training instances for which background phase was predicted
    :param targets_train_background: A 1D array of the targets for the background training instances
    :param train_rising: A 2D array of the training instances for which rising phase was predicted
    :param targets_train_rising: A 1D array of the targets for the rising training instances
    :param train_falling: A 2D array of the training instances for which falling phase was predicted
    :param targets_train_falling: A 1D array of the targets for the falling training instances
    :param algorithm: regular or rnn
    :return: The three trained models
    """

    # Set up identical structures for all three networks
    background_model = Sequential()
    if algorithm == 'regular':
        background_model.add(Dense(30, activation='sigmoid', input_shape=train_background.shape[1:]))
    else:
        background_model.add(GRU(30, activation='sigmoid', input_shape=train_background.shape[1:],
                                 return_sequences=False))
    background_model.add(Dense(1))
    rising_model = clone_model(background_model)
    falling_model = clone_model(background_model)
    background_model.compile(loss='mse', optimizer='adam')
    rising_model.compile(loss='mse', optimizer='adam')
    falling_model.compile(loss='mse', optimizer='adam')

    # Train networks
    background_model.fit(train_background, targets_train_background, epochs=1000, verbose=1,
                         callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    rising_model.fit(train_rising, targets_train_rising, epochs=1000, verbose=1,
                     callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    falling_model.fit(train_falling, targets_train_falling, epochs=1000, verbose=1,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20)])
    return background_model, rising_model, falling_model


def predict_intensity(phase_selection_model, background_model, rising_model, falling_model, test, algorithm):
    model_series = []
    if algorithm == 'regular':
        test_phase = np.array([instance.flatten() for instance in test])
    else:
        test_phase = np.array([instance.T for instance in test])
    phases = phase_selection_model.predict(test_phase).argmax(axis=1)
    for i, test_instance in enumerate(test):

        # Depending on predicted phase, use corresponding intensity model to predict intensity
        if phases[i] == BACKGROUND:
            model_series.append(-8)
        elif phases[i] == FALLING:
            model_series.append(-7)
        else:
            model_series.append(-6)

    # Reshape test set depending on algorithm
    if algorithm == 'regular':
        test = np.array([instance.flatten() for instance in test])
    else:
        test = np.array([instance.T for instance in test])

    # Using indices, predict from each of the three models
    model_series = np.array(model_series)
    predictions_intensity = np.zeros(len(test))  # Initialize list and populate with each model's predictions
    ind_background = np.where(model_series == -8)
    predictions_intensity[ind_background] = background_model.predict(test[ind_background]).flatten()
    ind_falling = np.where(model_series == -7)
    predictions_intensity[ind_falling] = falling_model.predict(test[ind_falling]).flatten()
    ind_rising = np.where(model_series == -6)
    predictions_intensity[ind_rising] = rising_model.predict(test[ind_rising]).flatten()

    return np.array(predictions_intensity), model_series


def evaluate_intensity_models(targets_test, predictions, model_series, event_times, data, path, display):

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
        outfile = open(f'{path}/results_intensity.txt', 'w')
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
    parser.add_argument('-lm', '--load_model_selector', action='store_true', required=False,
                        default=False, help="Whether or not to load trained phase selection model")
    parser.add_argument('-lp', '--load_predictions_phase', action='store_true', required=False,
                        default=False, help="Whether or not to load phase predictions")
    parser.add_argument('-lmi', '--load_models_intensity', action='store_true', required=False,
                        default=False, help="Whether or not to load trained intensity models")
    parser.add_argument('-lpi', '--load_predictions_intensity', action='store_true', required=False,
                        default=False, help="Whether or not to load intensity predictions")
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
    load_model_selector = args.load_model_selector
    load_predictions_phase = args.load_predictions_phase
    load_models_intensity = args.load_models_intensity
    load_predictions_intensity = args.load_predictions_intensity
    path = args.path
    display = args.display
    seed = args.seed

    ########################################################################################################
    # Step 1: Use ML to select the model (phase) to be used for predicting intensity
    ########################################################################################################

    # Pair inputs and outputs for phase selection, and split into train/test
    # _s for switching/selection
    data = pd.read_csv('../Data/data.csv')
    train_s, targets_train_s, test_s, targets_test_s =\
        pair_input_output_switch(data, use_phase_inputs, prediction_time)

    # Reshape train/test depending on algorithm
    if algorithm == 'regular':
        train_s = np.array([instance.flatten() for instance in train_s])
        test_s = np.array([instance.flatten() for instance in test_s])
    else:
        train_s = np.array([instance.T for instance in train_s])
        test_s = np.array([instance.T for instance in test_s])

    # Either load predictions and model, load model and get predictions, or train model and get predictions
    if load_predictions_phase:
        if path:
            phase_selection_model = load_model(f"{path}/phase_selection_model")
            predictions_phase = load_2d(f"{path}/predictions_phase.txt")
        else:
            print("No path to load predictions from, exiting program.")
            exit(0)
    else:
        if load_model_selector:
            if path:
                phase_selection_model = load_model(f"{path}/phase_selection_model")
            else:
                print("No path to load model from, exiting program.")
                exit(0)
        else:

            sample_weights = find_sample_weights(targets_train_s)
            phase_selection_model = train_phase_selection_model(train_s, targets_train_s, sample_weights, algorithm)
            if path:
                phase_selection_model.save(f"{path}/phase_selection_model")

        # Predict with phase selection model, and save predictions
        predictions_phase = phase_selection_model.predict(test_s)
        if path:
            write_2d(f"{path}/predictions_phase.txt", predictions_phase)

    # Evaluate phase selection model
    evaluate_phase_selection_model(targets_test_s, predictions_phase, path)

    ########################################################################################################
    # Step 2: Use predicted models to predict intensity
    ########################################################################################################

    # _i for intensity
    train_i, targets_train_i, test_i, targets_test_i =\
        pair_input_output_intensity(data, use_phase_inputs, prediction_time)

    # Reshape training set depending on algorithm
    if algorithm == 'regular':
        train_i = np.array([instance.flatten() for instance in train_i])
    else:
        train_i = np.array([instance.T for instance in train_i])

    # Split training set based on predicted phase
    train_background, targets_train_background, train_rising, targets_train_rising, train_falling,\
        targets_train_falling = split_train_intensity(phase_selection_model, train_i, targets_train_i)

    # Either load predictions and model, load model and get predictions, or train model and get predictions
    if load_predictions_intensity:
        if path:
            predictions_intensity = load_series(f"{path}/predictions_intensity.txt")
            model_series = load_series(f"{path}/model_series.txt")
        else:
            print("No path to load predictions from, exiting program.")
            exit(0)
    else:
        if load_models_intensity:
            if path:
                background_model = load_model(f"{path}/background_model")
                rising_model = load_model(f"{path}/rising_model")
                falling_model = load_model(f"{path}/falling_model")
            else:
                print("No path to load model from, exiting program.")
                exit(0)
        else:
            background_model, rising_model, falling_model =\
                train_intensity_models(train_background, targets_train_background, train_rising, targets_train_rising,
                                       train_falling, targets_train_falling, algorithm)
            if path:
                background_model.save(f"{path}/background_model")
                rising_model.save(f"{path}/rising_model")
                falling_model.save(f"{path}/falling_model")

        # Predict with phase selection model, and save predictions
        predictions_intensity, model_series =\
            predict_intensity(phase_selection_model, background_model, rising_model,
                              falling_model, test_i, algorithm)
        if path:
            write_file(f"{path}/predictions_intensity.txt", predictions_intensity)
            write_file(f"{path}/model_series.txt", model_series)

    # Load events for evaluation
    event_file = open('../Data/event_timestamps.txt', 'r')
    lines = event_file.readlines()
    event_times = [line.split() for line in lines]
    event_file.close()

    # Select only events which are in the test set
    event_times_test = []
    first_test_event_time = list(targets_test_i.keys())[0]
    for event in event_times:
        if event[0] >= first_test_event_time:
            event_times_test.append(event)

    # Evaluate predictions
    target_times = list(targets_test_i.keys())
    predictions = {target_times[i]: predictions_intensity[i] for i in range(len(predictions_intensity))}
    model_series = {target_times[i]: model_series[i] for i in range(len(model_series))}
    evaluate_intensity_models(targets_test_i, predictions, model_series, event_times_test, data, path, display)


if __name__ == "__main__":
    main()
