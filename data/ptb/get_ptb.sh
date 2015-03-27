#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget stewart.guru/s/ptb/vocab.pkl
wget stewart.guru/s/ptb/train_indices.txt
wget stewart.guru/s/ptb/valid_indices.txt
wget stewart.guru/s/ptb/test_indices.txt

echo "Done."
