# coding=UTF-8

# python train.py [training_data] [input_model_name]

import sys
from DNN import learn

#learn(training_data, input_model_name)
if len(sys.argv) == 2:
	learn(sys.argv[1])
elif len(sys.argv) == 3:
	learn(sys.argv[1], sys.argv[2])

