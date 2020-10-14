#!/bin/bash

set -e
set -x

find . -name "*.pyc" -type f -delete

python3.7 -m pip install --upgrade pip
python3.7 -m pip install -r requirements.txt

for DEVPKG in data_layer moda subir
do
  cd $DEVPKG
  if [ -f requirements.txt ]; then
    python3.7 -m pip install -r requirements.txt
  fi
  python3.7 setup.py develop
  cd ..
done
