import pandas as pd
import json

def get_dict():
    count_dict = {}

    fake_text = pd.read_csv('FakeTrain.csv')
    true_text = pd.read_csv('TrueTrain.csv')

    all_text = fake_text['title'].fillna("").tolist() + fake_text['text'].fillna("").tolist()
    all_text += true_text['title'].fillna("").tolist() + true_text['text'].fillna("").tolist()

    punctuations = [' ', '.', ',', ';', ':', '(', ')', '!', '?', '"', "'", '#', '\\']
    for string in all_text:
        lowercase = string.lower()
        actual_word = ''
        for i in range(len(lowercase)):
            if lowercase[i] in punctuations:
                actual_word += ' '
            else:
                actual_word += lowercase[i]
        lower_list = actual_word.split()
        for word in lower_list:
            if word not in count_dict:
                count_dict[word] = 0
            count_dict[word] += 1


    vocab_dict = {}
    vocab_dict['<PAD>'] = 0
    vocab_dict['<UNK>'] = 1
    current_index = 2
    for word in count_dict:
        if count_dict[word] >= 5:
            vocab_dict[word] = current_index
            current_index += 1
    
    return vocab_dict

def main():
    vocab_dict = get_dict()
    with open('vocab_dict.json', 'w') as file:
        json.dump(vocab_dict, file, indent=4)

if __name__ == '__main__':
    main()

