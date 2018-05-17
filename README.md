# PyUltraLight
PyUltraLight Repository


# Prerequisites

In order to run the Jupyter notebook, a number of Python packages must first be installed. These include matplotlib, numba, pyfftw, and h5py. These can be installed from the command line in ubuntu as follows:

###### matplotlib
sudo apt-get install python-matplotlib

###### numba
sudo apt-get install zlib1g-dev llvm-3.5-dev
LLVM_CONFIG=/usr/bin/llvm-config-3.5 sudo -E pip2 install llvmlite
sudo pip2 install enum34funcsigs numba

###### pyfftw
sudo apt-get install libfftw3-dev libfftw3-doc
sudo pip2 install pyfftw

###### h5py
sudo pip2 install h5py	
