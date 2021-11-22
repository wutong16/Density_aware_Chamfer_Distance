#!/bin/bash

pip install -r requirements.txt

cd utils/ChamferDistancePytorch/chamfer3D/
python setup.py install

cd ../../emd
python setup.py install

cd ../Pointnet2.PyTorch/pointnet2
python setup.py install

cd ../../../



