#RNNこれくしょん~RNNこれ~

**RNN-colle**はRNN (Recurrent Neural Networks)を用いた機械学習ライブラリです。GPUによる高速化も可能。Reservoir ComputingやDeep LSTMといった最先端のRNNモデルで学習を行ったり、オリジナルのRNNを定義するのも簡単にできます。
ライセンス：apache
行列計算, GPUにmshadowを使用
jsonのパースにpicojsonを使用

### 特徴
- **最新のRNNモデルを実装** - Reservoir computing, LSTM, etc.
- **mshadowベースの実装** - CPU/GPUの切り替えが容易に可能
- json formatのパラメタファイルで簡単に学習を管理
- オリジナルのRNNアーキテクチャを定義することも可能

### 要件
- CBLAS or MKL
- C++ compiler (g++, icc, etc.)
- python
 - numpy
 - wget
- (optional) nvidia GPU & CUDA 6.5

### Build
まず、`fetch.sh`を実行する。これで必要なデータがfetchされる。次に、
- GPUを用いるとき
`make gpu=1`
- 行列計算にMKLを用いるとき
`make mkl=1`
- 両方
`make gpu=1 mkl=1`

### Document
[Tutorial(jp)](https://github.com/mattya/RNN-colle/wiki/Tutorial_jp)

### Contributors
- [@mattya1089][1]
- [Dwango AI Lab.][2]
- [Kaneko lab. @ the University of Tokyo][3]


[1]:https://twitter.com/mattya1089
[2]:http://ailab.dwango.co.jp/
[3]:http://chaos.c.u-tokyo.ac.jp/index_j.html