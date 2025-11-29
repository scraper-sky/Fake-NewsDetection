import pandas as pd

FOLDER_PATH = "NewsDataset/"

def main():
    true_path = FOLDER_PATH + "True.csv"
    fake_path = FOLDER_PATH + "Fake.csv"
    true_data = pd.read_csv(true_path)
    fake_data = pd.read_csv(fake_path)
    
    true_data['label'] = 1
    fake_data['label'] = 0
    
    true_data['combined'] = true_data['title'] + " " + true_data['text']
    fake_data['combined'] = fake_data['title'] + " " + fake_data['text']
    
    true_processed = true_data[['combined', 'label']].sample(frac=1, random_state=229)
    fake_processed = fake_data[['combined', 'label']].sample(frac=1, random_state=229)
    
    true_processed_path = FOLDER_PATH + "true_processed.csv"
    fake_processed_path = FOLDER_PATH + "fake_processed.csv"
    true_processed.to_csv(true_processed_path, index=False)
    fake_processed.to_csv(fake_processed_path, index=False)
    
    return
    
if __name__ == "__main__":
    main()