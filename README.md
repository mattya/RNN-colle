#RNN collection

**RNN collection** is RNN (Recurrent Neural Networks) & Deep Learning library.

### Feature
- **Collection of state-of-the-art RNN models** - Deep LSTM, Reservoir Computing, etc.
- **mshadow based implementation** - you can easily switch cpu/gpu calculation
- json format parameter file
- supports define original RNN model

### Prerequisites
- CBLAS or MKL
- C++ compiler (g++, icc, etc.)
- python
 - numpy
 - wget
- (optional) nvidia GPU & CUDA 6.5

### Build
run `fetch.sh` and then
`make`
- If you want to use GPU
`make gpu=1`
- If you have MKL
`make mkl=1`
- Both
`make gpu=1 mkl=1`

### Getting started
`./bin/train < ./param_files/narma.json > log.txt`

### Document
[Tutorial(jp)](https://github.com/mattya/RNN-colle/wiki/Tutorial_jp)

### Backbone Library
- [mshadow](https://github.com/dmlc/mshadow)
- [picojson](https://github.com/kazuho/picojson)

### Contributors
- [@mattya1089][1]
- [Dwango AI Lab.][2]
- [Kaneko lab. @ the University of Tokyo][3]


[1]:https://twitter.com/mattya1089
[2]:http://ailab.dwango.co.jp/
[3]:http://chaos.c.u-tokyo.ac.jp/index_j.html