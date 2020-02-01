#!/bin/bash

set -e
set -x

if [ -e data/$SEASON ]; then
    echo "Directory data/$SEASON already exists!"
    exit 1
fi

[ -d data ] || mkdir data
cd data

DATASET_VERSION=V4

FILENAME=LADD_${DATASET_VERSION}_${SEASON}.zip

if [ ! -e $FILENAME ]; then
    if [ -z $LADD_BASEURL ]; then
        echo "Download $FILENAME or specify LADD_BASEURL"
        exit 1
    fi
    wget $LADD_BASEURL/$FILENAME
fi

unzip $FILENAME

title() {
    sed 's/^\(.\)/\U\1/g' <<< $1
}

mv "LizaAlertDroneDataset${DATASET_VERSION}_$(title $SEASON)" $SEASON
