# NLP-Caffe

NLP-Caffe is a pull request on the Caffe framework developed by Yangqing Jia and Evan Shelhamer, other members of the BVLC lab at Berkeley, and online contributers. This fork makes it easier for NLP users to get started without merging C++ code. Currently, there is an example of a language model for the Penn Tree Bank using LSTMs that processes 20,000 words per second. More examples for Machine Translation using the encoder-decoder model and character-level RNNs will be forthcoming. Eventually this code should be merged into the Caffe master branch.

# Installation

Please consult the Caffe <a href="http://caffe.berkeleyvision.org/installation.html">installation instructions</a> first. After successfully installing caffe, just clone this repo and run `make -j8` from the NLP-Caffe folder.
