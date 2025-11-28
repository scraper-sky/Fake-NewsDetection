import pandas as pd
import os
from sklearn.model_selection import train_test_split

def process_liar():
    data_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 60)
    print("PROCESSING LIAR DATASET")
    print("=" * 60)
    
    train_path = os.path.join(raw_dir, 'train.tsv')
    valid_path = os.path.join(raw_dir, 'valid.tsv')
    test_path = os.path.join(raw_dir, 'test.tsv')
    
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found!")
        print("Please download LIAR dataset and extract to raw/ directory")
        print("Expected files: train.tsv, valid.tsv, test.tsv")
        return
    
    print(f"\n1. Loading LIAR dataset...")
    
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
               'barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true', 'context']
    
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=columns)
    valid_df = pd.read_csv(valid_path, sep='\t', header=None, names=columns) if os.path.exists(valid_path) else pd.DataFrame()
    test_df = pd.read_csv(test_path, sep='\t', header=None, names=columns) if os.path.exists(test_path) else pd.DataFrame()
    
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(f"   Loaded {len(all_df)} statements")
    
    print(f"\n2. Mapping labels...")
    true_labels = ['true', 'mostly-true', 'half-true']
    fake_labels = ['false', 'barely-true', 'pants-fire']
    
    all_df['is_true'] = all_df['label'].apply(
        lambda x: 1 if x in true_labels else (0 if x in fake_labels else None)
    )
    all_df = all_df[all_df['is_true'].notna()]
    
    fake_df = all_df[all_df['is_true'] == 0].copy()
    true_df = all_df[all_df['is_true'] == 1].copy()
    
    print(f"   Fake: {len(fake_df)} statements")
    print(f"   True: {len(true_df)} statements")
    
    fake_df['text'] = fake_df['statement']
    fake_df['title'] = fake_df['statement'].str[:100]
    true_df['text'] = true_df['statement']
    true_df['title'] = true_df['statement'].str[:100]
    
    fake_df = fake_df[fake_df['text'].notna() & (fake_df['text'] != '')]
    true_df = true_df[true_df['text'].notna() & (true_df['text'] != '')]
    
    print(f"\n3. After removing empty:")
    print(f"   Fake: {len(fake_df)} statements")
    print(f"   True: {len(true_df)} statements")
    
    print(f"\n4. Creating train/test split (80/20)...")
    fake_train, fake_test = train_test_split(
        fake_df, test_size=0.2, random_state=42
    )
    true_train, true_test = train_test_split(
        true_df, test_size=0.2, random_state=42
    )
    
    print(f"   Fake - Train: {len(fake_train)}, Test: {len(fake_test)}")
    print(f"   True - Train: {len(true_train)}, Test: {len(true_test)}")
    
    print(f"\n5. Saving split files...")
    fake_train[['title', 'text']].to_csv(os.path.join(processed_dir, 'FakeTrain.csv'), index=False)
    fake_test[['title', 'text']].to_csv(os.path.join(processed_dir, 'FakeTest.csv'), index=False)
    true_train[['title', 'text']].to_csv(os.path.join(processed_dir, 'TrueTrain.csv'), index=False)
    true_test[['title', 'text']].to_csv(os.path.join(processed_dir, 'TrueTest.csv'), index=False)
    
    print(f"   ✓ Saved to {processed_dir}/")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Generate vocab: cd src/data && python3 generate_dict.py")
    print("2. Generate IDs: cd src/data && python3 create_vector_dataset.py")
    print("=" * 60)

if __name__ == '__main__':
    process_liar()

