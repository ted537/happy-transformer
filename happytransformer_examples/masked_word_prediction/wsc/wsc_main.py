
# from happytransformer import HappyROBERTA
from happytransformer import HappyBERT
from happytransformer import HappyXLNET
import pandas as pd
from happytransformer_examples.masked_word_prediction.wsc.data_collection_wsc import test_generator







def main():
    # test_generator("wsc_278.csv")
    wsc_278 = pd.read_csv("wsc_278.csv")


    happy_roberta = HappyXLNET("xlnet-large-cased")

    correct_count = 0
    total_count = 0
    for test_case_row in wsc_278.itertuples():

        sentence = test_case_row[1]
        OptionA = test_case_row[2]
        OptionB = test_case_row[3]
        correct_answer= test_case_row[4]


        options = [OptionA, OptionB]

        result = happy_roberta.predict_mask(sentence, options)[0]
        text_result = result["word"]
        if text_result == OptionA:
            if correct_answer == "A":
                correct_count += 1

        elif text_result == OptionB:
            if correct_answer == "B":
                correct_count += 1

        total_count += 1
        print("Total:", total_count, "\nCorrect:", correct_count, "\nPercentage:", (correct_count / total_count) * 100,
              "%")

    #
    # print("Final: ")
    # print("Total:", total_count, "\nCorrect:", correct_count, "\nPercentage:", (correct_count/ total_count)* 100, "%")
    #
    #

if __name__ == "__main__":
     main()

