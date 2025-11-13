import pandas as pd
import json

punctuations = [' ', '.', ',', ';', ':', '(', ')', '!', '?', '"', "'", '#', '\\']

def get_list(string):
    lowercase = string.lower()
    actual_word = ''
    for i in range(len(lowercase)):
        if lowercase[i] in punctuations:
            actual_word += ' '
        else:
            actual_word += lowercase[i]
    lower_list = actual_word.split()
    return lower_list

def get_maximum(data_path):
    data = pd.read_csv(data_path)
    title_len = 0
    text_len = 0
    for index, row in data.iterrows():
        title_list = get_list(row['title'])
        if len(title_list) > title_len:
            title_len = len(title_list)
        
        text_list = get_list(row['text'])
        if len(text_list) > text_len:
            text_len = len(text_list)
    return title_len, text_len

def get_id(string, vocab_dict, max_len):
    result = []
    array = get_list(string)
    for item in array:
        key = '<UNK>'
        if item in vocab_dict:
            key = item
        result.append(vocab_dict[key])

    while len(result) < max_len:
        key = '<PAD>'
        result.append(vocab_dict[key])
    
    return result


def create_json(data_path, save_path, vocab_dict, subject_dict, title_len, text_len, label_val):
    result = pd.DataFrame(columns=['title_id', 'text_id', 'subject_id', 'label'])

    data = pd.read_csv(data_path)
    vocab_columns = ['title', 'text']
    subject_columns = ['subject']

    for index, row in data.iterrows():
        title_id = get_id(row['title'], vocab_dict, title_len)
        text_id = get_id(row['text'], vocab_dict, text_len)
        
        subject_key = '<UNK>'
        if row['subject'] in subject_dict:
            subject_key = row['subject']
        subject_id = subject_dict[subject_key]

        result.loc[index] = [title_id, text_id, subject_id, label_val]
    
    result.to_csv(save_path, index=False)
    return

def main():
    with open('vocab_dict.json', 'r') as file:
        vocab_dict = json.load(file)
    with open('subject_dict.json', 'r') as file:
        subject_dict = json.load(file)
    
    title_len, text_len = get_maximum('TrainingData.csv') 

    create_json('FakeTrain.csv', 'FakeIDTrain.csv', vocab_dict, subject_dict, title_len, text_len, 0)
    create_json('FakeTest.csv', 'FakeIDTest.csv', vocab_dict, subject_dict, title_len, text_len, 0)
    create_json('TrueTrain.csv', 'TrueIDTrain.csv', vocab_dict, subject_dict, title_len, text_len, 1)
    create_json('TrueTest.csv', 'TrueIDTest.csv', vocab_dict, subject_dict, title_len, text_len, 1)

if __name__ == '__main__':
    main()