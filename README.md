
This repository corresponds to the work "A Machine Learning Approach to Predicting SEP Proton Intensity and Events Using Time Series of Relativistic Electron Measurements", to be published to Space Weather.


### Notes on Differences from Manuscript
Some metrics and models have been renamed, as the content of the manuscript has changed over time. These changes are described as follows:
- The manuscript only describes two neural network models, M1 and M3. However, there were two different multiple-model approaches used in initial tests, M3-MT (3 models selected with manual thresholds), and M3-ML (3 models selected with machine learning). Therefore, there are 3 neural network-based models in this repository. The M3 approach in the manuscript corresponds to M3-ML in this repository.
- Earlier versions of the work tracked three different lags: the lag calculated between the onset and peak (O2P lag), the lag calculated between the onset and threshold (O2T lag), and the lag calculated at the time the proton flux crosses the ln10 threshold (ln10 lag). Only the O2P lag is reported in the manuscript, and is referred to simply as "lag".


### Data
- _SO_expave15m.dat_ contains the 5-minute source data for electron and proton integral flux (http://www2.physik.uni-kiel.de/SOHO/phpeph/EPHIN.htm). This file needs to be unzipped before being used. Running the _data_preprocess.py_ script converts this data into the file _data.csv_, which is used by many scripts in this repository.
- _p10f10event.dat_ contains the list of SEP events. Running _event_analysis.py_ converts this data into _event_indices.py_ and _event_timestamps.py_, which are used by many scripts in this repository.
- _xray_1995_2002_5m_filled.txt_ contains the 5-minute source data for x-ray intensity (https://www.ngdc.noaa.gov/stp/satellite/goes/dataaccess.html). _data_interpolated_observed.csv_ contains the data from _data.csv_ combined with the xs and xl fields of the x-ray intensity data, with the timestamps interpolated to those of the x-ray data. _data_interpolated_derived.csv_ contains features derived from the source x-ray data, such as normalization, integral, and natural logarithm of x-ray intensity. The two .csv files are used by the scripts which take x-ray features as input. _data_interpolated_derived.csv_ needs to be unzipped before being used.


### Main Python scripts

- _persistent_model.py_ is in the top-level directory since it uses neither electron nor x-ray inputs. It is run with the following:
```python persistent_model.py [-h] [-t PREDICTION_TIME] [-p PATH] [-d]```

 - > options:
  -h, --help
  show this help message and exit
  -t PREDICTION_TIME, --prediction_time PREDICTION_TIME
   Number of timesteps ahead to predict
  -p PATH, --path PATH
  Directory to load and store files
  -d, --display
  Whether or not to display event plots


- The _ElectronInput_ directory contains 5 main scripts: 3 for the neural network-based models, one for the method from Posner (2007), and one for the task of classifying SEP events based on continuous proton flux predictions from the aforementioned models.
    - _m1.py_ and _m3_mt.py_ have identical command line arguments, and are run using the following:
```{m1.py, m3_mt.py} [-h] [-t PREDICTION_TIME] [-a {regular,rnn}] [-ph] [-lm] [-lp] [-p PATH] [-d] [-s SEED]```
 - > options:
  -h, --help
      show this help message and exit
  -t PREDICTION_TIME, --prediction_time PREDICTION_TIME
      Number of timesteps ahead to predict
  -a {regular,rnn}, --algorithm {regular,rnn}
      Algorithm to be used for intensity model (regular or RNN)
  -ph, --phase_inputs
      Whether or not to use phase inputs
  -lm, --load_model
      Whether or not to load trained model
  -lp, --load_predictions
      Whether or not to load predictions
  -p PATH, --path PATH
      Directory to load and store files
  -d, --display
      Whether or not to display event plots
  -s SEED, --seed SEED
      Seed for random number generator
    - _m3_ml.py_ has a slightly different usage from the other two neural network-based models since it has models for both selection and intensity. It is run using the following:
```m3_ml.py [-h] [-t PREDICTION_TIME] [-a {regular,rnn}] [-ph] [-lm] [-lp] [-lmi] [-lpi] [-p PATH] [-d] [-s SEED]```

   - > options:
    -h, --help
    show this help message and exit
    -t PREDICTION_TIME, --prediction_time PREDICTION_TIME
    Number of timesteps ahead to predict
    -a {regular,rnn}, --algorithm {regular,rnn}
    Algorithm to be used for both model selection and intensity models (regular or RNN)
    -ph, --phase_inputs
    Whether or not to use phase inputs
    -lm, --load_model_selector
     Whether or not to load trained phase selection model
    -lp, --load_predictions_phase
     Whether or not to load phase predictions
    -lmi, --load_models_intensity
     Whether or not to load trained intensity models
    -lpi, --load_predictions_intensity
     Whether or not to load intensity predictions
    -p PATH, --path PATH
    Directory to load and store files
    -d, --display
    Whether or not to display event plots
    -s SEED, --seed SEED
    Seed for random number generator
	- _posner_method.py_ is run similarly to the above methods, but only uses the arguments for prediction time, path, and display
	_time_series_classification.py_ takes a list of arguments from sys.argv rather than argparse, so no flags are used. Therefore, all arguments are required, and must be in the correct order. The arguments are as follows:
 - The first argument is the prediction time, 6 or 12
 - The second argument is the path, which must be a directory containing run1, run2, run3, run4, and run5 directories, each of these containing _predictions.txt_ output from one of the neural network approaches
 - The third argument is the classification approach name which must be one of the following:
       - "app0" corresponding to Approach W from the manuscript
       - "app1" corresponding to Approach EW
       - "app2" corresponding to Approach EAW
       - "app3" corresponding to Approach AW

- The _ElectronAndXrayInput_  directory contains 3 main scripts, one for each neural network-based approach. They are run similarly to the scripts in the _ElectronInput_ directory, but have an additional command line argument for x-ray features. The flag is -f, or --xray\_feature, and the string following the flag must be the name of a column from _data_interpolated_derived.csv_.

### Outputs
- Each of the main scripts outputs a model (except for the persistent model), a set of predictions, a set of results, and a plot for each SEP event in the test set. The outputs are saved to a user-specified path. For neural network-based approaches, if running multiple times, it is recommended to name the directories run1, run2, run3, run4, and run5, all in the same parent directory. The output directory must exist before running a script which will output to that directory.
- For the neural network-based methods, the model can have no extension, or a .keras or .h5 extension; this depends on the version of keras being used. File extensions will need to be changed as needed in the scripts on lines containing model.save() or model.load(). For Posner's method, the model is not saved directly since it does not take long to run and is deterministic; however, an image of the forecasting matrix is saved.
- For the neural network-based methods and persistent model, the predictions are a .txt file, with a line for each 5-minute timestep in the test set (whether or not an SEP event is occurring). For Posner's method, the predictions are saved in .csv format, as well as the training targets for use with external plotting scripts.
- For all methods, the results are saved in .txt format, containing the metrics for each individual SEP event in the test set, as well as averaged across all SEP events in the test set. Additionally, M3-ML saves results for the selector model.


### Supporting scripts
- _average_results.py_ - Standalone script to average metrics across multiple runs. It takes one command line argument, a directory containing subdirectories run1, run2, run3, run4, and run5, each of which contains a _results.txt_ file (or in the case of M3-ML, a _results_intensity.txt_ file). The script is currently hardcoded to average results from exactly 5 runs, which was the experimental procedure used for the manuscript.
- _load_data.py_ - Helper functions for saving and loading data.
- _stats.py_ - Helper functions for metrics tracked by models, including MAE and lag.
