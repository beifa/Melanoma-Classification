#!/bin/bash
echo 'Script start eff'
python3 train.py -e 50 -m eff -f 0 -l FocalLoss
sleep 15
python3 train.py -e 50 -m eff -f 1 -l FocalLoss
sleep 15
python3 train.py -e 50 -m eff -f 2 -l FocalLoss
sleep 15
python3 train.py -e 50 -m eff -f 3 -l FocalLoss
sleep 15
python3 train.py -e 50 -m eff -f 4 -l FocalLoss