'''
This script runs a simple machine-learning example.

It is the code found on slide 11 of 'week7_improving network part 2.pdf'

Modifications from example 3
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

net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, lmbda = 0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
