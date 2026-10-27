# src/utils/pira_model.py
from src.utils.utils import predict_over_time, pad_sequence, predict_model, predict_model_window, load_gpu, load_args, load_model_scaler
from timesnet.data_provider.uea import Normalizer
from timesnet.features import PIRA_FEATURES
from pathlib import Path

class EMCarePredictor:
    def __init__(self, args_path, model_path, scaler_path):
        self.args = load_args(args_path)
        self.args = load_gpu(self.args)
        model_path = Path(model_path).resolve()
        scaler_path = Path(scaler_path).resolve()
        #self.model, self.scaler = self._load_model_and_scaler(model_path, scaler_path)
        self.model, self.scaler = self._load_model_and_scaler(
            model_path,
            scaler_path
        )
        self.features = list(PIRA_FEATURES)  # [FIX B2] fuente única en timesnet/features.py

   # def _load_model_and_scaler(self, model_file, scaler_file):
    #    model, scaler = load_model_scaler(model_file, scaler_file, self.args)
     #   return model, scaler

    def _load_model_and_scaler(self, model_file, scaler_file):
        model, scaler = load_model_scaler(str(model_file), str(scaler_file), self.args)
        return model, scaler

    def predict(self, df):
        return predict_model(self.model, self.scaler, self.args, df, self.features)

    def predict_over_time(self, df):
        return predict_over_time(self.model, self.scaler, self.args, df, self.features)

    def predict_window(self, df, seq_len=None):
        return predict_model_window(self.model, self.args, df, self.scaler, seq_len=seq_len)
    
    def pad_sequence(self, df, method="zeros", position="pre"):
        return pad_sequence(df, self.args.seq_len, self.features, method, position)