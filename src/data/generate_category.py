import pandas as pd
import json

def get_dict():
    count_dict = {}

    fake_text = pd.read_csv('FakeTrain.csv')
    true_text = pd.read_csv('TrueTrain.csv')

    all_text = fake_text['subject'].fillna("").tolist() 
    all_text += true_text['subject'].fillna("").tolist() 

    for string in all_text:
        lowercase = string.lower()
        if lowercase not in count_dict:
            count_dict[lowercase] = 0
        count_dict[lowercase] += 1


    subject_dict = {}
    subject_dict['<PAD>'] = 0
    subject_dict['<UNK>'] = 1
    current_index = 2
    for word in count_dict:
        if count_dict[word] >= 5:
            subject_dict[word] = current_index
            current_index += 1
    
    return subject_dict

def main():
    subject_dict = get_dict()
    with open('subject_dict.json', 'w') as file:
        json.dump(subject_dict, file, indent=4)

if __name__ == '__main__':
    main()