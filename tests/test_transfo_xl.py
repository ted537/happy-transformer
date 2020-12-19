from happytransformer import HappyTransfoXL

happy = HappyTransfoXL()
happy.init_sequence_classifier()
happy.train_sequence_classifier('tests/train.csv')
happy.eval_sequence_classifier('tests/train.csv')