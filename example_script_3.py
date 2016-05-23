'''
This script runs a very simple machine-learning example.

It is the code found on slide 27 of 'Week6_Chapter 3 Improving Neural Networks.pdf'

The only modification in this code compared to example 2 is we've increased the number
of nuerons in layer two from 30 to 100. This modification is commented in the code.
'''

# This line is added so that python will use files found in the 'finech_task_dependencies' folder

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'finech_task_dependencies'))

import mnist_loader

import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# The value 100 denotes the number of layer two nuerons.
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)

net.large_weight_initializer()

net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

