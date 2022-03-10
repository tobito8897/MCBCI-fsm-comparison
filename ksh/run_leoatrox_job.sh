#!/bin/bash
source venv/bin/activate
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../bin`
cd $SCRIPTPATH$DATAPATH

/lustre/home/ssanchez/python-core_375/bin/python3 calculate_generic_windows_parameters_chb-mit.py --feature_set=2
deactivate
