import numpy as np
import pandas as pd


def find_indices():
    """
    Find the indices and timestamps for each event's onset, threshold, peak, and end.
    :return: Nothing; results are saved to a file, and this function only needs to be run once
    """

    # Open the two files, plus a third to output indices for use in multiple_models.py
    event_file = open("p10f10event.dat", "r")
    data_file = open("SO_expave15m.dat", "r")
    data_df = pd.read_csv("data.csv")

    # Convert each to a list of lines so that indices can be kept track of
    event_lines = event_file.readlines()
    event_lines = [event_lines[i].strip() for i in range(len(event_lines))][3:]
    data_lines = data_file.readlines()
    data_lines = [data_lines[i].strip() for i in range(len(data_lines))][7:]

    # Indices to keep track of
    onset_indices = []
    threshold_indices = []
    peak_indices = []
    end_indices = []
    event_index = 0

    # For each event, search data file for onset timestamp
    while event_index < len(event_lines):
        event_line = event_lines[event_index].split()
        onset_time = event_line[0]

        # Scan through the data until the timestamp is matched, or the end of the data is reached
        # This will search through the entire data file for each event, which is slow, but will not miss any events
        onset_found = False
        data_index = 0
        while data_index < len(data_lines) and not onset_found:
            data_line = data_lines[data_index].split()

            # The onset of the event is found
            if data_line[0] == onset_time:
                onset_found = True
                onset_indices.append(data_index)

                # Now find threshold time
                event_index += 1
                event_line = event_lines[event_index].split()
                threshold_time = event_line[0]

                # Match the threshold time in the data file
                threshold_found = False
                while data_index < len(data_lines) and not threshold_found:
                    data_line = data_lines[data_index].split()

                    # The threshold of the event is found
                    if data_line[0] == threshold_time:
                        threshold_found = True
                        threshold_indices.append(data_index)

                        # Now find peak time
                        event_index += 1
                        event_line = event_lines[event_index].split()
                        peak_time = event_line[0]

                        # Match the peak time in the data file
                        peak_found = False
                        while data_index < len(data_lines) and not peak_found:
                            data_line = data_lines[data_index].split()

                            # The peak of the event is found
                            if data_line[0] == peak_time:
                                peak_found = True
                                peak_indices.append(data_index)

                                # Finally, find end time
                                event_index += 1
                                event_line = event_lines[event_index].split()
                                end_time = event_line[0]

                                # Same matching procedure as before
                                end_found = False
                                while data_index < len(data_lines) and not end_found:
                                    data_line = data_lines[data_index].split()

                                    # The end of the event is found
                                    if data_line[0] == end_time:
                                        end_found = True
                                        end_indices.append(data_index)

                                        # Go to onset of next event
                                        event_index += 1

                                    # Check next data line (end)
                                    else:
                                        data_index += 1

                            # Check next data line (peak)
                            else:
                                data_index += 1

                    # Check next data line (threshold)
                    else:
                        data_index += 1

            # Check next data line (onset)
            else:
                data_index += 1

    # Save indices to a file
    out_file = open("event_indices.txt", "w")
    out_file_times = open("event_timestamps.txt", "w")
    for i in range(len(peak_indices)):
        if i != 15:
            out_file.write(f"{onset_indices[i]} {threshold_indices[i]} {peak_indices[i]} {end_indices[i]}\n")
            out_file_times.write(f"{data_df['time'].iloc[onset_indices[i]]} "
                                 f"{data_df['time'].iloc[threshold_indices[i]]} "
                                 f"{data_df['time'].iloc[peak_indices[i]]} "
                                 f"{data_df['time'].iloc[end_indices[i]]}\n")
    out_file.close()


def txt_to_df():

    infile = open("SO_expave15m.dat", "r")
    lines = infile.readlines()[7:]
    lines = [line.split() for line in lines]
    infile.close()

    for i in range(len(lines)):
        lines[i] = [lines[i][0]] + [int(lines[i][1])] + [float(value) for value in lines[i][2:]]

    features = ["electron", "electron_high", "proton", "proton_high"]
    columns = ["Julian_day", "Year", "DOY"]
    for feature in features:
        columns += [feature, f"{feature}_error", f"{feature}_fitted", f"{feature}_fitted_error",
                    f"d_{feature}", f"d_{feature}_error"]

    df = pd.DataFrame(lines, columns=columns)

    # Convert dates to a standard format
    df["time"] = (pd.to_datetime(df["Year"], format="%Y") +
                  pd.TimedeltaIndex(df["DOY"], unit="D") -
                  pd.Timedelta(1, unit="D"))
    df["time"] = [str(np.datetime64(time)) for time in df["time"]]

    df.to_csv("data.csv", header=True, index=False)


def add_phases():

    # Get event dates
    infile = open("event_timestamps.txt", "r")
    events = infile.readlines()
    events = [event.split() for event in events]
    infile.close()

    # Scan through dates in data, finding where each becomes background/rising/falling
    data = pd.read_csv(open("data.csv"))
    dates = data["time"].values
    event_index = 0
    event_phase = 0
    background = []
    rising = []
    falling = []
    date_to_find = events[event_index][event_phase]
    for date in dates:

        # Depending on current phase, shift to next phase/event index
        if date == date_to_find:

            # Background - go to peak
            if event_phase == 0:
                event_phase += 2

            # Peak - go to end
            elif event_phase == 2:
                event_phase += 1

            # End - go to onset of next event (skip event 15, which is a subset of event 16)
            else:
                event_phase = 0
                if event_index == 14:
                    event_index += 2
                else:
                    event_index += 1

            # Update date to find
            # If there are no more events, the rest is background
            if event_index < len(events):
                date_to_find = events[event_index][event_phase]

        # Add background/rising/falling labels for current timestamp
        if event_phase == 0:
            background.append(1)
            rising.append(0)
            falling.append(0)
        elif event_phase == 2:
            background.append(0)
            rising.append(1)
            falling.append(0)
        else:
            background.append(0)
            rising.append(0)
            falling.append(1)

    # Add to dataframe and filter columns; this is the final version of the dataframe used in other scripts
    data = data[["time", "electron", "d_electron", "electron_high", "d_electron_high", "proton", "d_proton"]]
    data["background"] = background
    data["rising"] = rising
    data["falling"] = falling
    data.to_csv("data.csv", header=True, index=False)


def main():
    txt_to_df()
    find_indices()
    add_phases()


if __name__ == "__main__":
    main()
