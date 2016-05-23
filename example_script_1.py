'''
This script runs a very simple machine-learning example.

It is the code found on slide 6 of 'Week4_Basic_Python_and_Neural_Network_Part2.pdf'

'''

# This line is added so that python will use files found in the 'finech_task_dependencies' folder

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'finech_task_dependencies'))

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

