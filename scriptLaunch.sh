#!/bin/bash

git pull origin
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
sudo /usr/bin/python3 $BASEDIR/scriptPython.py & >> scriptPython.log
