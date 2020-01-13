''' This module organizes example WS problems from cs.nyu.edu'''
import xml.etree.ElementTree as et
import pandas as pd
import re


def get_data_wsc():
    " Organizes the data_wsc into a panda dataframe"
    tree = et.parse('WSCollection.xml')

    root = tree.getroot()

    columns = ["txt1", 'pron', 'txt2', 'quote1', 'quote2', 'OptionA', 'OptionB', 'answer']

    dataframe = pd.DataFrame(columns=columns)

    for schema in root:
        problem = dict()
        for element in schema:
            i = 0
            for value in element:

                if value.tag == 'txt1':
                    problem['txt1'] = value.text

                elif value.tag == 'pron':
                    problem['pron'] = value.text

                elif value.tag == 'txt2':
                    problem['txt2'] = value.text

                elif value.tag == 'quote1':
                    problem['quote1'] = value.text

                elif value.tag == 'quote2':
                    problem['quote2'] = value.text

                elif value.tag == 'answer' and i == 0:

                    problem['OptionA'] = value.text
                    i = i + 1

                elif value.tag == 'answer' and i == 1:
                    problem['OptionB'] = value.text

                elif value.tag == "answer":

                    answer = value.text
                    answer = answer.strip()  # Remove whitespace
                    answer = answer[0]  # Remove possible "." trailing value
                    problem['answer'] = answer

        answer = schema.find("correctAnswer").text
        answer = answer.strip()  # Remove whitespace
        answer = answer[0]  # Remove possible "." trailing value

        problem['answer'] = answer
        dataframe = dataframe.append(problem, ignore_index=True)
    return dataframe


def test_generator(output_csv):
    """
    Uses the wsc 278 to generate testing examples that can be used with fitBERT.
    fitBERT requires that the masked word be labeled "[MASK]"
    The output is saved to a csv file called "fit_bert_test_generator"
    """

    df = get_data_wsc()

    masked_sentences = list()
    for index, row in df.iterrows():
        masked_sentence = row['txt1'] + " [MASK] " + row['txt2']
        masked_sentence = masked_sentence.replace("\n", " ")
        masked_sentence = re.sub(' +', ' ', masked_sentence)
        masked_sentences.append(masked_sentence)

    df["masked_sentences"] = masked_sentences
    new_df = df[['masked_sentences', 'OptionA', 'OptionB', "answer"]].copy()

    new_df.to_csv(output_csv, index=None, header=True)

    print("dataset successfully saved to ", output_csv)