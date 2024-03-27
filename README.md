## Evaluation of feature selection methods for classification of epileptic seizure EEG signals

### Overview

* This repository contains code used for the research https://doi.org/10.3390/s22083066.
* The configuration and metadata files are included too.
* Datasets are not included.

### Datasets

* CHB-MIT Scalp EEG Database is available at https://physionet.org/content/chbmit/1.0.0/ (accessed on 18 October 2021)
* Siena Scalp EEG Database is available at https://physionet.org/content/siena-scalp-eeg/1.0.0/ (accessed on 18 October 2021)

### Setup
You need to run a set of script before training the ML models. These will allow to download, clean and parametrize the datasets:

* CHB-MIT
```
ksh/create_tree_dir.sh
bin/download_edf_chb-mit.py
bin/collect_metadata_chb-mit.py
bin/delete_no_compliant_channels_chb-mit.py
bin/calculate_generic_windows_parameters_chb-mit.py
```

* Siena
```
create_tree_dir.sh
download_siena.sh
collect_metadata_siena.py
calculate_generic_windows_parameters_siena.py
```

Note: Some scripts require the user to provide arguments.

### Citing

If you find this repository useful, please consider citing our work.
```
@article{sanchez2022evaluation,
                title={Evaluation of feature selection methods for classification of epileptic seizure EEG signals},
                author={Sánchez-Hernández, Sergio E and Salido-Ruiz, Ricardo A and Torres-Ramos, Sulema and  Román-Godínez, Israel},
                journal={Sensors},
                volume={22},
                number={8},
                year={2022},
                publisher={MDPI},
                doi={https://doi.org/10.3390/s22083066}
}
```