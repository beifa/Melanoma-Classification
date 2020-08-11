#!/bin/bash
echo 'Script start res50'
# python3 train.py -e 50 -m res50 -f 0 
# sleep 15
python3 train.py -e 50 -m res50 -f 1 -l FocalLoss  
sleep 15
python3 train.py -e 50 -m res50 -f 2 -l FocalLoss  
sleep 15
python3 train.py -e 50 -m res50 -f 3 -l FocalLoss  
sleep 15
python3 train.py -e 50 -m res50 -f 4 -l FocalLoss  