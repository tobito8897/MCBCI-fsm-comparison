#!/bin/bash
set -e
SCRIPTPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATAPATH=`echo /../..`
cd $SCRIPTPATH$DATAPATH

mkdir -p "Data/ChbmitRaw"
mkdir -p "Data/SienaRaw"

mkdir -p "processeddata/ProcessedDataChbMit_fs1/Common/Ictal"
mkdir -p "processeddata/ProcessedDataChbMit_fs1/Common/NoIctal"

mkdir -p "processeddata/ProcessedDataChbMit_fs2/Common/Ictal"
mkdir -p "processeddata/ProcessedDataChbMit_fs2/Common/NoIctal"

mkdir -p "processeddata/ProcessedDataChbMit_fs1/GridSearch/Ictal"
mkdir -p "processeddata/ProcessedDataChbMit_fs1/GridSearch/NoIctal"

mkdir -p "processeddata/ProcessedDataChbMit_fs2/GridSearch/Ictal"
mkdir -p "processeddata/ProcessedDataChbMit_fs2/GridSearch/NoIctal"

mkdir -p "processeddata/ProcessedDataSiena_fs1/Common/Ictal"
mkdir -p "processeddata/ProcessedDataSiena_fs2/Common/NoIctal"

mkdir -p "mldata/MLDataChbmit_fs1/Correlation"
mkdir -p "mldata/MLDataChbmit_fs2/Correlation"

mkdir -p "mldata/MLDataSiena_fs1/Correlation"
mkdir -p "mldata/MLDataSiena_fs2/Correlation"

mkdir -p "mldata/MLDataChbmit_fs1/GridSearch"
mkdir -p "mldata/MLDataChbmit_fs2/GridSearch"

mkdir -p "mldata/MLDataChbmit_fs1/TopFeatures"
mkdir -p "mldata/MLDataChbmit_fs2/TopFeatures"

mkdir -p "mldata/MLDataSiena_fs1/TopFeatures"
mkdir -p "mldata/MLDataSiena_fs2/TopFeatures"

mkdir -p "mldata/MLDataChbmit_fs1/Stats"
mkdir -p "mldata/MLDataChbmit_fs2/Stats"

mkdir -p "mldata/MLDataSiena_fs1/Stats"
mkdir -p "mldata/MLDataSiena_fs2/Stats"