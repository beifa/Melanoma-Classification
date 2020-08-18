#!/bin/bash
echo 'Script start eff_3'
python3 train.py -e 50 -m eff3 -f 0 -s 300 -opt adam -shl slr
sleep 15
python3 train.py -e 50 -m eff3 -f 1 -s 300 -opt adam -shl slr
sleep 15
python3 train.py -e 50 -m eff3 -f 2 -s 300 -opt adam -shl slr
sleep 15
python3 train.py -e 50 -m eff3 -f 3 -s 300 -opt adam -shl slr
sleep 15
python3 train.py -e 50 -m eff3 -f 4 -s 300 -opt adam -shl slr