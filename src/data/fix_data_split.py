"""
Script to fix data leakage and properly split Fake.csv and True.csv into train/test sets.
This removes duplicates and ensures no articles appear in both train and test.
"""
import pandas as pd
import os
import numpy as np

def remove_duplicates(df):
    """Remove duplicate articles based on title and text."""
    print(f"  Original rows: {len(df)}")
    df_clean = df.drop_duplicates(subset=['title', 'text'], keep='first')
    print(f"  After removing duplicates: {len(df_clean)} (removed {len(df) - len(df_clean)})")
    return df_clean

def fix_data_split():
    data_dir = 'src/data'
    
    print("="*60)
    print("FIXING DATA SPLIT - REMOVING LEAKAGE AND DUPLICATES")
    print("="*60)
    
    # Load original data
    print("\n1. Loading data...")
    fake = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
    true = pd.read_csv(os.path.join(data_dir, 'True.csv'))
    
    print(f"   Fake.csv: {len(fake)} rows")
    print(f"   True.csv: {len(true)} rows")
    
    # Remove duplicates
    print("\n2. Removing duplicates...")
    fake_clean = remove_duplicates(fake)
    true_clean = remove_duplicates(true)
    
    # Also check for duplicates based on title only (in case same article has slightly different text)
    print("\n   Removing duplicates by title (keeping first occurrence)...")
    fake_clean = fake_clean.drop_duplicates(subset=['title'], keep='first')
    true_clean = true_clean.drop_duplicates(subset=['title'], keep='first')
    print(f"   Fake after title dedup: {len(fake_clean)} rows")
    print(f"   True after title dedup: {len(true_clean)} rows")
    
    # Split data properly (ensuring no leakage)
    print("\n3. Creating train/test split (80/20)...")
    # Use numpy for reproducible random split
    np.random.seed(42)
    
    fake_shuffled = fake_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(fake_shuffled) * 0.8)
    fake_train = fake_shuffled[:split_idx].reset_index(drop=True)
    fake_test = fake_shuffled[split_idx:].reset_index(drop=True)
    
    true_shuffled = true_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(true_shuffled) * 0.8)
    true_train = true_shuffled[:split_idx].reset_index(drop=True)
    true_test = true_shuffled[split_idx:].reset_index(drop=True)
    
    print(f"   Fake - Train: {len(fake_train)}, Test: {len(fake_test)}")
    print(f"   True - Train: {len(true_train)}, Test: {len(true_test)}")
    
    # Verify no leakage
    print("\n4. Verifying no data leakage...")
    fake_train_titles = set(fake_train['title'].str.lower().str.strip())
    fake_test_titles = set(fake_test['title'].str.lower().str.strip())
    true_train_titles = set(true_train['title'].str.lower().str.strip())
    true_test_titles = set(true_test['title'].str.lower().str.strip())
    
    fake_leakage = fake_train_titles.intersection(fake_test_titles)
    true_leakage = true_train_titles.intersection(true_test_titles)
    
    if len(fake_leakage) > 0 or len(true_leakage) > 0:
        print(f"   ⚠ WARNING: Still found leakage!")
        print(f"      Fake leakage: {len(fake_leakage)}")
        print(f"      True leakage: {len(true_leakage)}")
    else:
        print("   ✓ No data leakage detected!")
    
    # Save cleaned split files
    print("\n5. Saving cleaned split files...")
    fake_train.to_csv(os.path.join(data_dir, 'FakeTrain.csv'), index=False)
    fake_test.to_csv(os.path.join(data_dir, 'FakeTest.csv'), index=False)
    true_train.to_csv(os.path.join(data_dir, 'TrueTrain.csv'), index=False)
    true_test.to_csv(os.path.join(data_dir, 'TrueTest.csv'), index=False)
    
    print("   ✓ Files saved:")
    print(f"     - FakeTrain.csv ({len(fake_train)} rows)")
    print(f"     - FakeTest.csv ({len(fake_test)} rows)")
    print(f"     - TrueTrain.csv ({len(true_train)} rows)")
    print(f"     - TrueTest.csv ({len(true_test)} rows)")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Regenerate vocab_dict.json and subject_dict.json")
    print("2. Regenerate ID files using create_vector_dataset.py")
    print("="*60)

if __name__ == '__main__':
    fix_data_split()

