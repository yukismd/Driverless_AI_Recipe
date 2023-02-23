"""
Use Japanese BERT model for DAI Transformer
https://huggingface.co/cl-tohoku/bert-base-japanese
"""

from h2oaicore.systemutils import config
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.transformers_nlp import BERTTransformer

_global_modules_needed_by_name = ["fugashi", "ipadic"]

MODEL_NAME = 'cl-tohoku/bert-base-japanese'

class BertBaseJapanese(BERTTransformer, CustomTransformer):
    _mojo = False

    @staticmethod
    def get_default_properties():
        return dict(col_type="text",
                    min_cols=1,
                    max_cols=1,
                    relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return dict(model_type=[MODEL_NAME],
                    batch_size=[config.pytorch_nlp_fine_tuning_batch_size],
                    seq_length=[config.pytorch_nlp_fine_tuning_padding_length]
                    )
