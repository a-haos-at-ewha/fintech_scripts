'''
This script runs a slightly more complicated machine-learning example.

It is the code found on slide 26 of 'Week6_Chapter 3 Improving Neural Networks.pdf'

'''

# This line is added so that python will use files found in the 'finech_task_dependencies' folder

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'finech_task_dependencies'))

import mnist_loader

import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

net.large_weight_initializer()

net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

