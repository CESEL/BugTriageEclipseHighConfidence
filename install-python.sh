#!/bin/bash
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python3-pip
sudo pip3 install virtualenv
virtualenv -p python3 virtualenv-python
