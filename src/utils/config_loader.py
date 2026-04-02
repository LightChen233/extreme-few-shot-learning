"""
配置加载器
"""
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k, default)
            if value is None:
                return default
        return value

    @property
    def input_cols(self):
        return self.get('data.input_cols', [])

    @property
    def target_cols(self):
        return self.get('data.target_cols', [])

    @property
    def output_meta(self):
        return self.get('data.outputs', [])

    @property
    def test_path(self):
        train = self.get('data.train_path', 'data/train.csv')
        return str(Path(train).parent / 'test.csv')
