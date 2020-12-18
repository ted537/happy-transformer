"""
HappyXLNET: a wrapper over PyTorch's XLNet implementation
"""

from transformers import (
    TransfoXLTokenizer,
    TransfoXLForSequenceClassification,
    TransfoXLLMHeadModel
)

from happytransformer.happy_transformer import HappyTransformer


class HappyTransfoXL(HappyTransformer):
    """
    Currently available public methods:
        XLNetLMHeadModel:
            1. predict_mask(text: str, options=None, k=1)
        XLNetForSequenceClassification:
            1. init_sequence_classifier()
            2. advanced_init_sequence_classifier()
            3. train_sequence_classifier(train_csv_path)
            4. eval_sequence_classifier(eval_csv_path)
            5. test_sequence_classifier(test_csv_path)

    """

    def __init__(self, model='transfo-xl-wt103'):
        super().__init__(model, "TRANSFOXL")
        self.mlm = None
        self.tokenizer = TransfoXLTokenizer.from_pretrained(model)
        # using TOKEN_X_TOKEN because transfo xl tokenizer rudely eats up brackets
        self.tokenizer.cls_token = 'TOKEN_CLS_TOKEN'
        self.tokenizer.sep_token = 'TOKEN_SEP_TOKEN'