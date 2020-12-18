from happytransformer.happy_transfo_xl import HappyTransfoXL

happy = HappyTransfoXL()
happy.init_sequence_classifier()
happy.train_sequence_classifier('tests/train.csv')
print(predictions)