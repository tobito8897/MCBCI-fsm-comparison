#!/bin/bash
SCRIPTPATH=`pwd`
DATAPATH=`echo /../../Data/SienaRaw`
cd $SCRIPTPATH$DATAPATH
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/
cd "./physionet.org/files/siena-scalp-eeg/1.0.0"
cp -rv * ../../../../
cd $SCRIPTPATH$DATAPATH
rm -r physionet.org