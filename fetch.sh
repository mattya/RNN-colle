#! /bin/bash

if [ ! -e mshadow ]; then
  echo "Fetch mshadow..."
  git clone https://github.com/tqchen/mshadow.git -b master
fi

if [ ! -e picojson ]; then
  echo "Fetch picojson..."
  git clone https://github.com/kazuho/picojson.git -b master
fi

cd picojson
make
cd ..

mkdir -p data/mnist
mkdir -p data/narma
mkdir -p data/recall

echo "Fetch mnist..."
curl -o ./data/mnist/train-images.idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o ./data/mnist/train-labels.idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o ./data/mnist/t10k-images.idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o ./data/mnist/t10k-labels.idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Unzip mnist..."
gunzip ./data/mnist/train-images.idx3-ubyte.gz
gunzip ./data/mnist/train-labels.idx1-ubyte.gz
gunzip ./data/mnist/t10k-images.idx3-ubyte.gz
gunzip ./data/mnist/t10k-labels.idx1-ubyte.gz

cd tools
echo "Generate narma30 task..."
python gen_narma.py
echo "Generate recall task..."
python gen_recall.py

