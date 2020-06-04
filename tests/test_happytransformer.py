import pathlib

from happytransformer import HappyBERT,HappyROBERTA,HappyXLNET

TRAIN_BINARY_SEQ_CSV = pathlib.Path(__file__).parent / 'example_seq_train.csv'
TEST_BINARY_SEQ_CSV = pathlib.Path(__file__).parent / 'example_seq_test.csv'

transformers = [HappyBERT(),HappyROBERTA(),HappyXLNET()]

def test_maskedword():
    for transformer in transformers:
        predictions = transformer.predict_mask("Humans are for [MASK].")
        prediction = predictions[0]
        assert(
            isinstance(prediction['word'],str) and 
            isinstance(prediction['softmax'],float)
        )

def test_binary_seq():
    for transformer in transformers:
        transformer.init_sequence_classifier()
        transformer.train_sequence_classifier(TRAIN_BINARY_SEQ_CSV)
        eval_dict = transformer.eval_sequence_classifier(TRAIN_BINARY_SEQ_CSV)
        assert(
            eval_dict['true_positive'] >=0 and
            eval_dict['true_negative'] >=0 and
            eval_dict['false_positive'] >=0 and
            eval_dict['false_negative'] >=0
        )
        transformer.test_sequence_classifier(TEST_BINARY_SEQ_CSV)

if __name__=="__main__":
    pass
    # test_maskedword()