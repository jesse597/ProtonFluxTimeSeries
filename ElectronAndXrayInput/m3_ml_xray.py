from keras.callbacks import Callback, EarlyStopping
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


BACKGROUND = 0
RISING = 1
FALLING = 2

class ReplaceNan(Callback):
    """
    Loop through weights of each layer and replace NaN with 10^6.
    """

    def on_epoch_end(self, epoch, logs=None):
        for i, layer in enumerate(self.model.layers):
            weights = layer.get_weights()
            if len(weights) > 0:

                # For each weight matrix (weights, bias, extra weights from GRU), check for NaN
                new_weights = []
                for weight in weights:

                    # Flatten then reshape since number of dimensions can differ between layers
                    shape_orig = weight.shape
                    weight = weight.flatten()
                    for j in range(len(weight)):
                        if np.isnan(weight[j]):
                            weight[j] = 10 ** 6
                    weight = weight.reshape(shape_orig)
                    new_weights.append(weight)

                # Replace weights
                self.model.layers[i].set_weights(new_weights)


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


def pair_input_output_switch(data_observed, data_derived, xray_switch, prediction_time):
    """
    Pair inputs and outputs, and split into training and testing sets.
    :param data_observed: A DataFrame object containing observed features
    :param data_derived: A DataFrame object containing derived features
    :param xray_switch: An array containing xray values for switching models, or None
    :param prediction_time: The number of timesteps to predict ahead
    :return: Training and test inputs and targets
    """

    time = list(data_observed['time'].values)
    electron = data_observed['electron'].values
    electron_high = data_observed['electron_high'].values
    proton = data_observed['proton'].values
    phase = np.array([data_derived['background'].values, data_derived['rising'].values,
                      data_derived['falling'].values]).T

    # Pair inputs and outputs
    x = []
    y = []
    for t in range(60, len(data_observed) - prediction_time):

        # Get current instance (past 5 hours plus current)
        x_curr = [electron[t - 60: t + 1], electron_high[t - 60: t + 1], proton[t - 60: t + 1]]
        if xray_switch is not None:
            x_curr.append(xray_switch[t - 60: t + 1])

        # Add to list of all instances
        x.append(x_curr)

        # Store output phase; include timestamps for splitting
        y.append(phase[t + prediction_time])

    # Split train/test
    # Timestamp for splitting depends on prediction time, but both are close to each other
    if prediction_time == 6:
        time_split = '2000-12-17T03:40:00.000'
    elif prediction_time == 12:
        time_split = '2000-12-17T03:45:00.000'

    split_index = list(time[60: len(time) - prediction_time]).index(time_split) - prediction_time

    # Targets do not need to keep timestamps in this case, since no plotting
    train = x[:split_index]
    targets_train = y[:split_index]
    test = x[split_index:]
    targets_test = y[split_index:]
    return np.array(train), np.array(targets_train), np.array(test), np.array(targets_test)


def find_sample_weights(targets_train):
    counts = np.bincount(targets_train.argmax(axis=1))  # Order is background, rising, falling
    ratios = [1, 10 * counts[0] / counts[1], counts[0] / counts[2]]
    sample_weights = []
    for label in targets_train:
        sample_weights.append(ratios[label.argmax()])
    return np.array(sample_weights)


def train_phase_selection_model(train, targets_train, sample_weights, algorithm):

    # Create phase selection model
    model = Sequential()
    if algorithm == 'regular':
        model.add(Dense(30, input_shape=train.shape[1:], activation='relu'))
    else:
        model.add(GRU(30, input_shape=train.shape[1:], activation='sigmoid', return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train and return models
    model.fit(train, targets_train, epochs=1000, sample_weight=sample_weights,
              callbacks=[EarlyStopping(monitor='loss', min_delta=1e-4, patience=20), ReplaceNan()])
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
    print(f"Rising precision = {precision: 0.4f}")
    print(f"Rising recall = {recall: 0.4f}")
    print(f"Rising F1 = {f1: 0.4f}")

    # Output to file as well, if a path is provided
    if path:
        outfile = open(f"{path}/results_selection.txt", "w")
        outfile.write("Rows are actual, columns are predicted\n")
        outfile.write("Order is background, rising, falling\n")
        outfile.write(str(mtx))
        outfile.write(f"\nRising precision = {precision: 0.4f}\n")
        outfile.write(f"Rising recall = {recall: 0.4f}\n")
        outfile.write(f"Rising F1 = {f1: 0.4f}\n")
        outfile.close()


def pair_input_output_intensity(data, xray_integral, prediction_time):
    """
    Pair inputs and outputs, and split into training and testing sets.
    :param data: A DataFrame object
    :param xray_integral: An array containing xray integral values, or None
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
        x_curr = []
        x_curr.append(electron[t - 60: t + 1])
        x_curr.append(electron_high[t - 60: t + 1])
        x_curr.append(proton[t - 60: t + 1])
        if xray_integral is not None:
            x_curr.append(xray_integral[t - 60: t + 1])

        # Add to list of all instances
        x.append(x_curr)

        # Store output; include timestamps
        y.append([time[t + prediction_time], proton[t + prediction_time]])

    # Split train/test
    # Timestamp for splitting depends on prediction time, but both are close to each other
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
    return np.array(train), (np.array(targets_train)[:, 1]).astype(np.float64), np.array(test), targets_test


def split_train_intensity(model_selector, train_s, train_i, targets_train_i):

    # Training sets for intensity prediction
    train_background = []
    targets_train_background = []
    train_rising = []
    targets_train_rising = []
    train_falling = []
    targets_train_falling = []

    # Based on model predicted for time t, add instance at time t to corresponding training set
    phases = model_selector.predict(train_s).argmax(axis=1)
    for t in range(len(train_s)):
        if phases[t] == BACKGROUND:
            train_background.append(train_i[t])
            targets_train_background.append(targets_train_i[t])
        elif phases[t] == FALLING:
            train_falling.append(train_i[t])
            targets_train_falling.append(targets_train_i[t])
        else:
            train_rising.append(train_i[t])
            targets_train_rising.append(targets_train_i[t])

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
        background_model.add(Dense(30, activation='relu', input_shape=train_background.shape[1:]))
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


def predict_intensity(phase_selection_model, background_model, rising_model, falling_model, test_s, test_i, algorithm):

    # predictions_intensity = []
    model_series = []
    phases = phase_selection_model.predict(test_s).argmax(axis=1)
    for i, test_instance in enumerate(test_i):

        # Depending on predicted phase, use corresponding intensity model to predict intensity
        if phases[i] == BACKGROUND:
            model_series.append(-8)
        elif phases[i] == FALLING:
            model_series.append(-7)
        else:
            model_series.append(-6)

    # Reshape test set depending on algorithm
    if algorithm == 'regular':
        test_i = np.array([instance.flatten() for instance in test_i])
    else:
        test_i = np.array([instance.T for instance in test_i])

    # Using indices, predict from each of the three models
    model_series = np.array(model_series)
    predictions_intensity = np.zeros(len(test_i))  # Initialize list and populate with each model's predictions
    ind_background = np.where(model_series == -8)
    predictions_intensity[ind_background] = background_model.predict(test_i[ind_background]).flatten()
    ind_falling = np.where(model_series == -7)
    predictions_intensity[ind_falling] = falling_model.predict(test_i[ind_falling]).flatten()
    ind_rising = np.where(model_series == -6)
    predictions_intensity[ind_rising] = rising_model.predict(test_i[ind_rising]).flatten()

    return np.array(predictions_intensity), model_series


def evaluate_intensity_models(targets_test, predictions, model_series, event_times, xray_switch, data, path, display):

    # First, get electron, high energy electron, and xray from data with timestamps for plots
    times = list(data["time"].values)
    electron = data["electron"].values
    electron_high = data["electron_high"].values

    # Convert to dictionaries
    electron_dict = {times[i]: electron[i] for i in range(len(electron) - len(targets_test), len(electron))}
    electron_high_dict = {times[i]: electron_high[i] for i in range(len(electron_high) - len(targets_test),
                                                                    len(electron_high))}
    if xray_switch is not None:
        xray_dict = {times[i]: xray_switch[i] for i in range(len(xray_switch) - len(targets_test), len(xray_switch))}

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
        if xray_switch is not None:
            xray_o2p = slice_dictionary_keys(xray_dict, bg_before_onset, peak)
        model_series_o2p = slice_dictionary_keys(model_series, bg_before_onset, peak)

        # Plot event
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('ln(Flux (/cc/s/sr))')
        ax1.plot(list(targets_o2p.values()), '-b', label='Actual proton')
        ax1.plot(list(predictions_o2p.values()), '-r', label='Predicted proton')
        ax1.plot(list(electron_o2p.values()), '-m', label='Electron')
        ax1.plot(list(electron_high_o2p.values()), '-y', label='High-energy electron')
        ax1.plot([np.log(10)] * len(targets_o2p.values()), '--k')
        if xray_switch is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Xray Flux')
            ax2.plot(list(xray_o2p.values()), '-k', label='X-ray')
        # ax1.plot(list(model_series_o2p.values()), '-g', label='Model')

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
        outfile = open(f'{path}/results_intensity.txt', 'w')
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
    parser.add_argument('-f', '--xray_switch_feature', type=str, required=False,
                        help="Xray feature to be used for switching models")
    parser.add_argument('-t', '--prediction_time', type=int, required=False,
                        default=6, help="Number of timesteps ahead to predict")
    parser.add_argument('-a', '--algorithm', type=str, required=False, default='regular',
                        choices=['regular', 'rnn'], help="Algorithm to be used for both model selection and "
                                                         "intensity models (regular or RNN)")
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
    xray_switch_feature = args.xray_switch_feature
    prediction_time = args.prediction_time
    algorithm = args.algorithm
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
    data_observed = pd.read_csv('../Data/data_interpolated_observed.csv')
    data_derived = pd.read_csv('../Data/data_interpolated_derived.csv')
    if xray_switch_feature:
        xray_switch = data_derived[xray_switch_feature].values
        xray_integral = data_derived['xs_norm'].values
    else:
        xray_switch = None
        xray_integral = None

    # _s for switching/selection
    train_s, targets_train_s, test_s, targets_test_s =\
        pair_input_output_switch(data_observed, data_derived, xray_switch, prediction_time)

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
        pair_input_output_intensity(data_observed, xray_integral, prediction_time)

    # Reshape training set depending on algorithm
    if algorithm == 'regular':
        train_i = np.array([instance.flatten() for instance in train_i])
    else:
        train_i = np.array([instance.T for instance in train_i])

    # Split training set based on predicted phase
    train_background, targets_train_background, train_rising, targets_train_rising, train_falling,\
        targets_train_falling = split_train_intensity(phase_selection_model, train_s, train_i, targets_train_i)

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
                              falling_model, test_s, test_i, algorithm)
        if path:
            write_file(f"{path}/predictions_intensity.txt", predictions_intensity)
            write_file(f"{path}/model_series.txt", model_series)

    # Load events for evaluation
    event_file = open('../Data/event_timestamps_interpolated.txt', 'r')
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
    evaluate_intensity_models(targets_test_i, predictions, model_series, event_times_test,
                              xray_switch, data_observed, path, display)


if __name__ == "__main__":
    main()
