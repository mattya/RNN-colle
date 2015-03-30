#RNNこれくしょん~RNNこれ~

**RNN-colle**はRNN (Recurrent Neural Networks)を用いた機械学習ライブラリです。GPUによる高速化も可能。Reservoir ComputingやDeep LSTMといった最先端のRNNモデルで学習を行ったり、オリジナルのRNNを定義するのも簡単にできます。
ライセンス：apache
行列計算, GPUにmshadowを使用
jsonのパースにpicojsonを使用

[TOC]

### 特徴
- **最新のRNNモデルを実装** - Reservoir computing, LSTM, ...
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

### Tutorial
#####Multi layer perceptron
多層パーセプトロンでMNISTを学習する。
MNISTのデータは**./data/mnist/**の中に、学習に用いるパラメタは**./param_files/mnist.json**に記載されている。
numpyが使えるpythonのパスが`python`で無い場合は、**mnist.json**のpythonの項目を書き換える必要がある。


では、学習を開始しよう。
> ./bin/train < ./param_files/mnist.json > log.txt

**log.txt**には、
> epoch: 1, train error: 0.056024, test error: 0.063052
> epoch: 2, train error: 0.037949, test error: 0.044077
> epoch: 3, train error: 0.024398, test error: 0.035643

のように、学習の進行が記録される。今回の設定では、errorの数値は**誤答率**で表される。10 epochごとに、学習したモデルの情報が**tmp/mnist_model_10.txt**といったファイルに記録される。30 epochもすれば、train errorは1%を切り、過学習状態となるであろう。

次に、この学習したモデルを用いて、テストデータ全てについて、各手書き画像がどの数字を表しているかを推定させてみよう。
> ./bin/train < ./param_files/mnist_predict.json

このコマンドを実行すると、**./tmp/mnist_predict.txt**に結果が出力される。結果の各行はそれぞれ、各テストデータに対するNNの最終層の出力を表す。これにsoftmax関数を通したものが、NNの予測した各数字に対する確率分布ということになる。

最後に、**mnist.json**のパラメタについて解説する。
```
"env": {
      "python": "python",  # pythonのパス
      "debug": false,      # debug mode
      "prefix": "mnist"    # 生成されるファイルのprefix
      "mode": "train"      # train or predict
 }
```
modeを"predict"にすると、テストデータの入力に対する推定出力をファイルに書き出すモードになる。

```
"network": {
	"type": "MLP",    # ネットワークの種類
	"param": {
	     "layers": 4,                     # 層の数
	     "neurons": [784, 500, 200, 10],  # 各層のニューロン数
	     "hidden_nl": "tanh",             # 中間層の非線形関数
	     "out_nl": "none",                # 最終層の非線形関数
	     "loss": "category",              # 誤差の計算法
	     "shuffle": "full_shuffle",       # trainの際のshuffle
	     "dropout": false                 # dropout
	}
}
```
**network**はネットワークの構造を指定する。上記のパラメタでは、以下の様な4層パーセプトロンが作られる。
```flow
in=>operation: Input(784)
h1=>operation: 500
h2=>operation: 200
out=>operation: Output(10)

in(right)->h1(right)->h2(right)->out
```

```
"data": {
      "type": "MNIST",         # データの格納形式
      "train_data": "./data/mnist/train-images.idx3-ubyte",
      "train_label": "./data/mnist/train-labels.idx1-ubyte",
      "test_data": "./data/mnist/t10k-images.idx3-ubyte",
      "test_label": "./data/mnist/t10k-labels.idx1-ubyte",
      "predict": "./tmp/mnist_predict.txt",
                               # 予測結果の出力先(modeがpredictのとき)
      "n_train": 60000,        # trainデータの数
      "n_test": 10000,         # testデータの数
      "n_x_data": 784,         # dataの次元数
      "n_x_label": 10          # labelの次元数
      "load_model": false,     # 学習済みのモデルから再開するか
 },
```
**data**ではtrain, testデータの入出力ファイル、ファイル形式、次元数を指定する。

```
"learning": {
     "n_time": 1,             # BPTTの逆伝播ステップ数
     "n_batch": 60,           # batchサイズ
     "load_model": true,      # 学習済みのモデルから再開するか
     "init_epoch": 10,        # 何番目のepochから再開するか
     "iter_per_epoch": 1000,  # 1回のepochで何batch処理するか
     "snapshot_interval": 10, # modelファイルを出力する頻度
     "max_epoch": 100,        # 何epoch回すか
     "sgd": "rmsprop",        # 学習則 momentum or rmsprop
     "momentum": 0.9,         # momentum
     "decay": 0.001,          # weight decay
     "base_lr": 0.0001,       # learning rateの初期値
     "lr_mult_interval": 10,  # learning rateを変化させる頻度
     "lr_mult": 0.75,         # learning rateの変化倍率
}
```
**learning**では学習時のパラメタを指定する。MNISTは時系列データではないので、**n_time**は1である。



---
#####Reservoir computing
RNNのひとつのアルゴリズムであるReservoir ComputingでNARMA30というタスクを学習しよう(**narma.json**)。NARMA30は次のような規則で生成される。vは入力であり、各時刻[0,0.5]の一様分布で生成される。この入力vに対するxが予測するべき出力である。今回のデータではk=30となっており、RNNは30ステップを超える長い時間相関を学習する必要がある。
```
    for i in range(1,tmax):
        for j in range(max(0,i-k), i):
            x[i] += 0.004*x[i-1]*x[j]
        x[i] += 0.2*x[i-1] + 0.01
        if i>=k:
            x[i] += 1.5*v[i]*v[i-k]
```

データはテキストファイルで、各行に、各時刻での入力データが数字で書かれている。
```
"network": {
     "type": "Reservoir_MLP",   # Reservoirの後ろにMLPを付けたもの
     "param": {
          "base_neurons": 200,  # Reservoirのニューロン数
          "scale": 0.95,        # Reservoirの時間スケール
          "layers": 4,          # MLPの層数
          "neurons": [200, 100, 100, 1],  # MLPのニューロン数
          "loss": "mse",        # 二乗誤差
          "shuffle": "single_series"  #  入力が一本の長い時系列
     }
}
"learning": {
     "n_time": 100,       # 1つのbatchで100ステップ連続で処理する
     "handover": true,    # batchを切り替えるとき、Reservoirを初期化しない
     "n_batch": 20,
     (以下略)
}
```
このパラメタで表されるネットワークの構造は、以下のようになる。
![Alt text](./Screen Shot 2015-03-31 at 5.22.33 AM.png)

Reservoir Computingは再帰結合の行列をランダム行列で初期化し、そのまま固定する。この部分は学習せずに固定することにより、RNNをBPTTで学習する際に生じるvanishing gradient problemを回避するのである。本ライブラリでは、固有値が**scale**になるような直交行列で初期化される。

学習開始
> ./bin/train < ./param_files/narma.json > log.txt

学習後の様子。
![Alt text](./figure_3.png)

---
#####Deep LSTM
自然言語処理などに用いられる階層LSTMを用いて、シンプルな記憶タスクを学習しよう(**recall.json**)。

![Alt text](./Screen Shot 2015-03-31 at 6.35.57 AM.png)

ひとつのLSTMは、HiddenニューロンとCellニューロン、そしてその間の結合からなる。Cellは記憶を担当するニューロンで、情報が書き込まれたら、次に別の情報が来るか、忘却信号が来るまで今の値をずっと保持するようにできている。HiddenニューロンはCellから情報を取り出し、出力を生成する。この際、1ステップ前のHiddenの値によりCellに書き込む・読み込む・忘却させる情報が決定される。

本ライブラリにおけるLSTMの詳しい動作は、以下の通りである(http://arxiv.org/abs/1410.4615より引用)。
![Alt text](./lstm.png)

さらに、このLSTMを階層状に重ねることで、RNNとLSTMを組み合わせたようなモデルを実現する。**recall.json**のパラメタでは、以下の様なネットワークが構築される。LSTMのHidden, Cellニューロンはそれぞれ30ニューロンずつとなる。
![Alt text](./Screen Shot 2015-03-31 at 6.26.45 AM.png)

```
"network": {
     "type": "Stacked_LSTM_MLP",
     "param": {
          "layers": 2,               # LSTMの層数
          "base_neurons": 30,        # 各LSTMの基準ニューロン数
          "layers2": 3,              # MLPの層数
          "neurons": [50, 20, 1],    # MLPのニューロン数
          "out_nl": "sigmoid"        # MLPの出力にsigmoid関数
          "loss": "nll",             # negative log likelihood
          "shuffle": "multiple_series",
     }
}
```


データはバイナリファイルで、n_time*n_x個のint型のデータが格納されている。

学習後
![Alt text](./figure_13.png)




### プログラムの構成
- Node
- Layer
- Network
- Main
- Parameter files
- IO
- Util

### Detailed usage
1. Preparing data
2. Choose Node
3. Choose NN
4. Parameter file
5. Learning
6. Using model file

### Make your own RNN
1. Define Node
2. Define NN
3. Original parameter
4. Build

### TBA
- Network definition by parameter files
- more sophisticated loss, shuffle type impl
- partial loading
- Image Node, Im2Col, Col2Im, Convolution layer
- NADE Layer
- Neural Turing Machine
- Q-Learning

### Contributors
- [@mattya1089][1]
- [Dwango AI Lab.][2]
- [Kaneko lab. @ the University of Tokyo][3]


[1]:https://twitter.com/mattya1089
[2]:http://ailab.dwango.co.jp/
[3]:http://chaos.c.u-tokyo.ac.jp/index_j.html