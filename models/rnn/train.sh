#!/usr/bin/env sh

./build/tools/caffe train --solver=models/rnn/solver.prototxt $@
