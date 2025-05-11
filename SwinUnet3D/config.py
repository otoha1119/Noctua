# config.py

class Config:
    def __init__(self):
        # ── データや出力先のパス
        self.data_path    = "/path/to/BraTS2021"
        self.TrainPath    = "./output/Brats2021"
        self.PredDataDir  = "./predictions/Brats2021"

        # ── 汎用ハイパーパラメータ
        self.batch_size   = 1
        self.learning_rate= 1e-4
        self.epochs       = 100
        self.num_classes  = 4     # BraTS は 4 クラス

