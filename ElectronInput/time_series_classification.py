from load_data import *
from m1 import pair_input_output
import numpy as np
import pandas as pd
import sys


prediction_time = int(sys.argv[1])  # 6 or 12
path = sys.argv[2]  # Directory containing run1-run5
app = sys.argv[3]
# app0 - approach W
# app1 - approach EW
# app2 - approach EAW
# app3 - approach AW
if app not in [f"app{i}" for i in range(4)]:
    print("Invalid app selection, exiting.")
    exit(0)

# Test targets - same for all 5 runs
data = pd.read_csv("../Data/data.csv")
train, targets_train, test, targets_test = pair_input_output(data, False, prediction_time)
targets_test_values = list(targets_test.values())
times = data["time"].values

# Load events for evaluation
event_file = open('../Data/event_indices.txt', 'r')
lines = event_file.readlines()
event_times = [line.split() for line in lines]
event_times = [[int(index) for index in event] for event in event_times]
event_file.close()

# Select only events which are in the test set
event_times_test = []
for event in event_times:
    if event[0] >= 24 + len(targets_train):
        event_times_test.append([event[1], event[3]])  # Threshold, end

# Evaluate for different alert windows
for alert_window in [1, 3, 6, 9, 12, 24, 72]:

    if alert_window < 12:
        print(f"Alert window = {alert_window * 5} minutes")
    elif alert_window == 12:
        print("Alert window = 1 hour")
    else:
        print(f"Alert window = {alert_window * 5 // 60} hours")

    tps = []
    fps = []
    fns = []

    # Evaluate predictions from each run
    for i in range(1, 6):
        # Collect list of alert durations (start index, end index)
        predictions = load_series(f"{path}/run{i}/predictions.txt")

        alert_durations = []
        t = 0
        while t < len(predictions):

            # Start alert duration
            if predictions[t] >= np.log(10):
                start = t + 24 + len(targets_train) + prediction_time

                # Blue bar
                if app in ["app2", "app3"]:
                    start -= prediction_time

                # Extend alert duration as needed, starting from last timestamp with intensity > ln10 (yellow/green bar)
                if app in ["app1", "app2"]:
                    while True in (predictions[t + 1: t + alert_window + 1] > np.log(10)):
                        for t2 in range(alert_window, 0, -1):
                            if predictions[t + t2] >= np.log(10):
                                t += t2
                                break
                else:

                    # Yellow bar without green bar
                    while predictions[t + 1] >= np.log(10):
                        t += 1

                end = t + alert_window + 24 + len(targets_train) + prediction_time  # 6 hours after t, offset by training set
                alert_durations.append([start, end])
                t += alert_window
                if app in ["app2", "app3"]:
                    t += prediction_time  # Prevent overlapping alerts

            else:
                t += 1

        # Check for TPs and FPs - for each alert duration, does it overlap the start of an SEP event,
        # is there no SEP event for the entire alert duration, or does the alert take place too late?
        tp = 0
        fp = 0
        fn = 0

        for j, event_duration in enumerate(event_times_test):

            overlap = False
            for alert_duration in alert_durations:
                if alert_duration[0] < event_duration[0] < alert_duration[1]:
                    overlap = True
                    tp += 1
                    break

            if not overlap:
                fn += 1

        for alert_duration in alert_durations:

            # Check for overlap with any SEP event, and determine which case
            overlap = False
            for event_duration in event_times_test:

                # Alert duration overlaps beginning of SEP event - TP, but it was already counted so just ignore
                if alert_duration[0] < event_duration[0] < alert_duration[1]:
                    overlap = True
                    break

                # Alert duration is entirely inside SEP event - not counted
                elif event_duration[0] < alert_duration[0] < alert_duration[1] < event_duration[1]:
                    overlap = True
                    break

                # Overlapping with end of event but not beginning; not counted
                elif event_duration[0] < alert_duration[0] < event_duration[1] < alert_duration[1]:
                    overlap = True
                    break

            # If alert duration does not overlap any SEP event, it is an FP
            if not overlap:
                fp += 1

        print(f"Run {i}: TP = {tp}, FN = {fn}, FP = {fp}")
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    print(f"Average: TP = {np.mean(tps):.1f}, FN = {np.mean(fns):.1f}, FP = {np.mean(fps):.1f}\n")

    # If not using green bar, no need to loop over remaining alert window sizes
    if alert_window == 1 and app in ["app0", "app3"]:
        exit(0)
