#!/bin/bash
echo 'Script start eff'
python3 train.py -e 50 -m eff2 -f 0 -l FocalLoss -s 260
sleep 15
python3 train.py -e 50 -m eff2 -f 1 -l FocalLoss -s 260
sleep 15
python3 train.py -e 50 -m eff2 -f 2 -l FocalLoss -s 260
sleep 15
python3 train.py -e 50 -m eff2 -f 3 -l FocalLoss -s 260
sleep 15
python3 train.py -e 50 -m eff2 -f 4 -l FocalLoss -s 260