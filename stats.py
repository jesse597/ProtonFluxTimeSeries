"""
Author: Jesse Torres
Description: This module implements functions for calculating
stats on the neural network's predictions, including RMSE,
chi-square error, TSS and F1 based on thresholds, and feature importance.
Additionally, a function for plotting actual vs. predicted is implemented.
"""

# from features import sigma
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np


def rmse(targets, predictions, error_function):
    """
    Calculates the RMSE of the predictions using the given error function.
    :param targets: A list of observed values of the output
    :param predictions: A list of predicted values of the output, same length as targets
    :param error_function: "intensity_error" or "percent_diff"
    :return: The RMSE of the predicted values
    """
    if error_function == "intensity_error":
        return np.sqrt(np.mean((np.e ** targets - np.e ** predictions) ** 2))
    elif error_function == "percent_diff":
        return np.sqrt(np.mean((10 ** predictions / 10 ** targets) - 1) ** 2)


def mae(targets, predictions, before_log):
    """
    Calculates the MAE of the predictions.
    :param targets: A list of observed values of the output
    :param predictions: A list of predicted values of the output
    :param before_log: A boolean indicating whether to undo the ln() function
    :return: The MAE of the predicted values
    """
    if before_log:
        return np.mean(np.abs(np.exp(targets) - np.exp(predictions)))
    else:
        return np.mean(np.abs(targets - predictions))


def x_axis_error(targets, predictions, before_log, display=False, path=None, event_index=None):
    """
    Calculate the lag between the targets and predictions.
    :param targets: A list of observed values of the output
    :param predictions: A list of predicted values of the output
    :param before_log: A boolean indicating whether to undo the ln() function
    :param display: Whether or not to show plots while running
    :param path: The path to which to save plots
    :param event_index: The index to use for plot filenames
    :return: The lag with the lowest MAE
    """
    maes = []
    for lag in range(min(len(targets), 24)):

        # Push targets forward to align with predictions, and cut off predictions so that sizes are the same
        targets_shifted = targets[:len(targets) - lag]
        predictions_adjusted = predictions[lag:]

        # Find the lag with the lowest MAE
        maes.append(mae(targets_shifted, predictions_adjusted, before_log))

    maes = np.array(maes)

    # Plot lag vs. MAE (and MAE - min(MAE)
    if path:
        min_mae = min(maes)
        plt.plot(np.arange(len(maes)), maes, 'ro', label='MAE')
        plt.plot(np.arange(len(maes)), maes - min_mae, 'bo', label='Difference from min MAE')
        plt.legend()
        plt.title("Lag vs. MAE")
        plt.xlabel("Lag")
        plt.ylabel("MAE")
        if path:
            plt.savefig(f"{path}/Lag_vs_mae_event{event_index}.png")
        if display:
            plt.show()
        plt.close()
    return maes.argmin()


def lag_ln10(targets, predictions):
    """
    Calculates the lag between when predictions first exceed ln10 and when targets first exceed ln10.
    :param targets: A list of observed values of the output
    :param predictions: A list of predicted values of the output
    :return: A single number which is the difference between where each list exceeds ln10
    """
    actual_ln10_time = -1  # Placeholder values for checking that both are found
    predicted_ln10_time = -1
    ln10 = np.log(10)
    for i in range(len(targets)):

        # If actual time has not been found yet, and exceeds threshold...
        if actual_ln10_time == -1 and targets[i] > ln10:
            actual_ln10_time = i

        # If predicted time has not been found yet, and exceeds threshold...
        if predicted_ln10_time == -1 and predictions[i] > ln10:
            predicted_ln10_time = i

        # If both have been found, no need to continue looking
        if actual_ln10_time != -1 and predicted_ln10_time != -1:
            break

    # If the predictions never exceed ln10, assign the index of peak
    if predicted_ln10_time == -1:
        print("Predictions do not exceed ln10, setting predicted threshold time to end of event")
        predicted_ln10_time = len(predictions) - 1

    # If for some reason the event never crosses the threshold, alert user (this should not happen)
    if actual_ln10_time == -1:
        print("Event never exceeds ln10")

    # Predicted - actual since predicted is expected to come after actual
    return predicted_ln10_time - actual_ln10_time


def lag_ln10_bool(targets, predictions):
    """
    Calculates lag between when targets and predictions exceed ln10, where the values in each list are y/n.
    :param targets: A list of observed classifications in the form 0 for no, 1 for yes
    :param predictions: A list of predicted classifications, same form as targets
    :return: The lag between the first predicted yes and first actual yes
    """
    actual_ln10_time = -1  # Placeholder values for checking that both are found
    predicted_ln10_time = -1
    for i in range(len(targets)):

        # If actual time has not been found yet, and exceeds threshold...
        if actual_ln10_time == -1 and targets[i] == 1:
            actual_ln10_time = i

        # If predicted time has not been found yet, and exceeds threshold...
        if predicted_ln10_time == -1 and predictions[i] == 1:
            predicted_ln10_time = i

        # If both have been found, no need to continue looking
        if actual_ln10_time != -1 and predicted_ln10_time != -1:
            break

    # If the predictions never exceed ln10, set time to last index for now
    if predicted_ln10_time == -1:
        print("ln10 not predicted, setting time to last index")
        predicted_ln10_time = len(predictions) - 1

    # If for some reason the event never crosses the threshold, alert user
    if actual_ln10_time == -1:
        print("Event never exceeds ln10")

    # Predicted - actual since predicted is expected to come after actual
    return predicted_ln10_time - actual_ln10_time


def tss_f1(targets, predictions):
    """
    Calculate TSS and F1 using above 1 PFU (before log, 0 after log)
    as positive class, below 1 PFU before log as negative
    :param targets: A list of observed values of the output
    :param predictions: A list of predicted values of the output
    :return: The TSS and F1 score of the predicted values
    """

    # Discretize values based on SEP threshold
    targets = np.array(targets)
    targets_bool = targets > 0
    predictions_bool = predictions > 0

    # Find confusion matrix and use values to calculate TSS and F1 score
    mtx = confusion_matrix(targets_bool, predictions_bool)
    print(f"Confusion matrix:\n{mtx}")
    tn, fp, fn, tp = mtx.ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    f1 = f1_score(targets_bool, predictions_bool)
    return tss, f1


def calc_feature_importance(estimator, features):
    """
    Calculate the relative importance of each input feature.
    :param estimator: The trained neural network (using sklearn MLPRegressor;
                      this will not work with Keras GRU)
    :param features: The list of feature names
    :return: Nothing
    """

    # Get weights and normalize along columns
    weights = estimator.coefs_
    for mtx in weights:
        for j in range(mtx.shape[1]):
            column = mtx[:, j]
            column = np.abs(column)
            column = column / sum(column)
            mtx[:, j] = column

    # Get importance values by multiplying weight matrices through each layer
    importance_values = weights[0]
    for j in range(1, len(weights)):
        importance_values = np.matmul(importance_values, weights[j])

    # Normalize result
    importance_values = importance_values / sum(importance_values)

    # Get feature names and pair with importance values
    feature_importance = {}
    for j in range(len(features)):
        feature_importance[f"{features[j]}"] = importance_values[j][0]

    # Sort and output feature importances
    # features_sorted only contains names of features, so they are accessed in the
    # dictionary in order of importance
    print("Feature Importances:")
    features_sorted = sorted(feature_importance, key=lambda col: feature_importance[col], reverse=True)
    for feature in features_sorted:
        print(f"{feature}: {feature_importance[feature]}")
    print()


def plot_predicted_vs_actual(targets, predictions, display=False, filename=None):
    """
    Plots predictions and observed values on the same plot.
    :param targets: The observed values of the outputs
    :param predictions: The values output from the neural network
    :param display: If provided, display the plot
    :param filename: If provided, save plot with given filename
    :return: Nothing
    """
    plt.plot(predictions, '-r', label='Predicted proton')
    plt.plot(targets, '-b', label='Actual proton')
    plt.legend()
    # plt.xlabel("Time (5-minute increments)")
    plt.ylabel("Intensity")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    if display:
        plt.show()
    plt.close()
