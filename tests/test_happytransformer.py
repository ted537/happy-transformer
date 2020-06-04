from happytransformer import HappyBERT,HappyROBERTA,HappyXLNET

transformers = [HappyBERT(),HappyROBERTA(),HappyXLNET()]

def test_maskedword():
    for transformer in transformers:
        predictions = transformer.predict_mask("Humans are for [MASK].")
        prediction = predictions[0]
        assert(
            isinstance(prediction['word'],str) and 
            isinstance(prediction['softmax'],float)
        )

if __name__=="__main__":
    pass
    # test_maskedword()