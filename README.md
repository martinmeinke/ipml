ipml
====

Use the IMPLDriver to run anything. Create a setup and configure what should be run, what should be loaded and what should be saved.
You should be able to configure *everything* through that setup.
Make sure your training data is located in the "data/" directory (not included here because of size).

How To generate features
------------------------
Make sure your training data is located under `data/`.
In the IMPLDriver, modify the numeric values for the run configuration of `generateAndSaveFeatures`.
`DataProviderMax` defines how many training pictures should be considered.
`num_features` defines the number of features.
`max_texel_pics` defines how many pictures should be used to create the texel feature vector.
The last value could be set to `5000` for a system with 8GB RAM and a SSD (so the system doesn't mind swapping).

Make sure the driver runs this `generateAndSaveFeatures` configuration. When you run the program, you
can observe the output in `logs/<current date>_main.log` (e.g. with `tail -f`).

Once finished, the generated data is pickeled as `data_segmentation` and `extracted_features` in the `saved/` directory.
You can change that location by providing an absolute path as `DataSavePath` and `FeatureSavePath` in the run configuration.
You might consider using the `IPMLRunConfiguration.PROJECT_BASEDIR` variable as an absolute path to the project directory.
Make sure these path are accessible, they are not checked in advanced.

Alternatively, you can just rename `data_segmentation` and `extracted_features` after creation and commit them to make them available
Suggested filenames: `<filename>.<numfiles>.<numfeatures>`


NUMPA
-----
How to install numpa on Ubuntu:
http://blogs.bu.edu/mhirsch/2014/07/installing-llvm-py-and-numba-on-ubuntu-14-04/


Tests
-----
run run_tests.py
