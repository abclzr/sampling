#!/bin/bash

cd torch2trt
python main.py -m vgg -s 10000
python main.py -m resnet -s 10000
python main.py -m inception -s 10000