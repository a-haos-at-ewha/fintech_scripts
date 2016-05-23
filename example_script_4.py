'''
This script runs a simple machine-learning example.

It is the code found on slide 33 of 'Week6_Chapter 3 Improving Neural Networks.pdf'

Modifications from example 3
We have limited the training data to 1000 samples.
We have requested 400 Epochs be ran.
We have turned on monitoring of training cost by setting it's value to true.
'''

# This line is added so that python will use files found in the 'finech_task_dependencies' folder

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'finech_task_dependencies'))

import mnist_loader

import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# The value 100 denotes the number of layer two nuerons.
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

net.large_weight_initializer()

net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
