B4M1勉強会2014 - High Performance Programming
====

概要
----
Fx10やMICを最大限利用した行列演算を作成し、競ってください。

内容
----
このテストプログラムをリンクし、測定を行います。
その際に、Fx10とMIC向けの設定項目が存在ので注意してください。(コンパイラやエンディアンが異なるため)

最も簡単な使い方
----
* 適当なx86やamd64のマシン(自分のラップトップなど)でg++によりビルドする場合
  1. srcの中にある、mymulmat.cppを変更する
  2. このREADME.mdがあるディレクトリに移動して、`make`コマンドを実行
  3. 生成されるmain.binを実行する
  4. 問題セットを変更したい場合は、data/list-intel.txtを変更する。

* FX10向けにビルドする場合
  1. Makefileを開き、PLATFORMの項目をFX10に変更する
  2. data/list.txtのシンボリックリンクを、data/list-fx10.txtに張り替える。
     dataディレクトリにcdした後、`ln -fs list-fx10.txt list.txt`を実行すればできる。
  3. srcの中にある、mymulmat.cppを変更する
  4. このREADME.mdがあるディレクトリに移動して、`make`コマンドを実行
  5. run.bashの中のrscgrp, node, elapseなどを適切に設定し、`pjsub run.bash`を実行
  6. `pjstat`コマンドでタスクが終わるのを確認する
  7. run.bash.oXXXXXXのようなファイルに結果が書き込まれているのを確認する
  8. 問題セットを変更したい場合は、data/list-fx10.txtを変更する。

* MIC向けにビルドする場合
  1. Makefileを開き、PLATFORMの項目をXEONPHIに変更する
  2. srcの中にある、mymulmat.cppを変更する
  3. このREADME.mdがあるディレクトリに移動して、`make`コマンドを実行
  4. 出来上がったmain.bin、及びdataディレクトリをmicに転送し、実行する。
  5. 問題セットを変更したい場合は、data/list-intel.txtを変更する。


また、main.binを始めとする副生成物を削除したい場合は、
```
make clean
```
を実行してください。

MPIを利用する際の注意
----
MPIを利用する際には、Makefileに以下の変更を加える必要があります。
1. USEMPIオプションを1にする
2. 利用するプラットフォームのCXXを変更する
   * MYLOCALであれば、g++からmpic++に変更する
   * FX10であれば、FCCpxからmpiFCCpxに変更する
   * XEONPHIであれば、iccからmpiiccに変更する

プログラムの全体の実行の流れとしては以下のようになるので注意してください。
1. [ALL]   MPI::Init()
2. [ALL]   mymulmatのコンストラクタ
3. [ALL]   mymulmat.init n,m,kは全てのプロセスに同じものが渡されます。
4. [RANK0] A, Bに値がセットされる
5. [ALL]   mymulmat.multiply A, Bのブロードキャストは各自で実行してください。
6. [RANK0] Cをテスト
7. [ALL]   mymulmatのデストラクタ
8. [ALL]   MPI::Finalize()

オプション
----
`./main.bin` : リスト中の<free>の中からランダムにファイルを取ってきて実行
`./main.bin -h` : ヘルプ
`./main.bin -v` : バージョン
`./main.bin -t type` : タイプを指定して実行。typeとしてはfree, mv(mat-vec), trmm(Aが上三角), symm(Aが対称), square(正方)がある。

自分のリポジトリとの結合
----
別にリポジトリを自分で作ってしまって、mymulmat.(cpp|h)をシンボリックリンクに置き換えてしまうのが手軽でよいと思います。

テスト
----
適当なデータセットでプログラムを走らせ、WRONGが0であればそのデータセットにおいてテストが通っていると言えます。

測定
----
正式な競技内容に関しては検討中です。
詳細が決まるのをお待ちください。

結果の参照
----
測定結果をうまく表示するためのシステムを用意しようと考えています。
実装をお待ちください。

作者
----
Makoto Shimazu <shimazu@eidos.ic.i.u-tokyo.ac.jp>
