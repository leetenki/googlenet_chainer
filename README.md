# GoogLeNetのChainerモデル
## 学習済みGoogLeNetを試す

以下のコマンドで、指定したURLの画像をダウンロードし、その画像の分類結果を出力する。

```
python googlenet_predict.py -u http://find-travel.cdn-dena.com/picture/articlebody/38214
```

`-u`オプションは省略可能。


## GoogLeNetのチューニング
まずは `tuned_googlenet.py` を編集してニューラルネットワークの構成を変える。CNNでは深層に行くほど画像の根本的な特徴量が抽出されるので、基本的に出力層に近いレイヤーだけ取り替えればOK。ここでは、全inception module層の学習済みパラメータを固定し、以下の層のみ取り替えて再学習を行わせる。

- 分岐点1のloss1\_conv層、loss1\_fc1層、loss1\_fc2層
- 分岐点2のloss2\_conv層、loss2\_fc1層、loss2\_fc2層
- 出力層のloss3\_fc

これらの層のlink及びユニット数などを解くべき問題に合わせて修正した後、以下のコマンドを実行すれば、必要最小限の学習済みのパラメータが設定されたgooglenetのモデルファイルが作られる。

```
python convert_caffe_model.py
```

設定済みのchainerモデルファイルは、`tuned_googlenet.model` という名前で出力される。ここの変換では具体的に、caffeのmodel zooにあるオリジナルの学習済みモデル(bvlc_googlenet.caffemodel)から、学習済みパラメータをchainer側のモデルにコピーしている。(チューニングするレイヤー以外)

あとは、 `train_googlenet.py` で、chainer側のモデルを読み込んで好きに学習させればOK。

※ちなみに、`tuned_googlenet.py` のモデル定義ファイルでは、誤差逆伝播をfc層で止めるために、__call__の順伝播処理内でvolatileフラグを使って計算履歴を削除している。volatile = Trueとすれば、そこから先の計算ではhistoryに残らず、backwardする際に逆伝播が止まる。volatile = Falseにすると、そこから先は順伝播のhistoryが残るので、backwardする際にWとbの勾配計算が行われる仕組み。