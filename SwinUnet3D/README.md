# SwinUnet3D
デモの最後の文字が1であれば普通に使用でき、0であれば画像の前処理が不十分で、効果が非常に悪く、サイコロ係数が0.001以下であることを意味します。

対応するデータセットをダウンロードした後、忘れずにConfigクラスで下図のdata_path、TrainPath、PredDataDirを自分のパスに変更してトレーニングを完了します。モデルの予測結果はPredDataDirに保存され、予測セットはiTK-SNAPまたは3Dslicerで開かれ、PredDataDirの対応するデータはiTK-SNAPまたは3Dslicerで開かれます。 iTK-SNAPまたは3Dslicerで予測セットを開き、PredDataDirの対応する予測結果をそこにドラッグして、セグメンテーション結果を3Dで視覚化します。
! [img.png](img.png)

マスター関数の位置をトレーニングします：
! [image](https://github.com/1152545264/SwinUnet3D/assets/44309924/701a2631-7561-4d86-a4fa-9bdec941318a)
他のデモも同じ

バージョン問題: V1は単純に変換器を使って実装されたもの、V2は変換器と畳み込みを混ぜた論文でのメインバージョン

もし私たちの研究があなたの役に立つなら、対応する論文を引用してください: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02129-z
