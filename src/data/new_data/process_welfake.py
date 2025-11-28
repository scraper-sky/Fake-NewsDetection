import pandas as pd
import os
from sklearn.model_selection import train_test_split

def process_welfake():
    data_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 60)
    print("PROCESSING WELFake DATASET")
    print("=" * 60)
    
    welfake_path = os.path.join(raw_dir, 'WELFake_Dataset.csv')
    
    if not os.path.exists(welfake_path):
        print(f"ERROR: {welfake_path} not found!")
        print("Please download WELFake_Dataset.csv and place it in the raw/ directory")
        return
    
    print(f"\n1. Loading {welfake_path}...")
    df = pd.read_csv(welfake_path)
    print(f"   Loaded {len(df)} articles")
    
    print(f"\n2. Dataset columns: {list(df.columns)}")
    
    if 'label' in df.columns:
        fake_df = df[df['label'] == 0].copy()
        true_df = df[df['label'] == 1].copy()
    elif 'Label' in df.columns:
        fake_df = df[df['Label'] == 0].copy()
        true_df = df[df['Label'] == 1].copy()
    else:
        print("ERROR: Could not find 'label' or 'Label' column")
        print("Available columns:", list(df.columns))
        return
    
    print(f"\n3. Split by label:")
    print(f"   Fake: {len(fake_df)} articles")
    print(f"   True: {len(true_df)} articles")
    
    fake_df = fake_df[fake_df['text'].notna() & (fake_df['text'] != '')]
    true_df = true_df[true_df['text'].notna() & (true_df['text'] != '')]
    
    print(f"\n4. After removing empty text:")
    print(f"   Fake: {len(fake_df)} articles")
    print(f"   True: {len(true_df)} articles")
    
    target_total = 20000
    target_per_class = target_total // 2
    
    if len(fake_df) > target_per_class:
        print(f"\n4.5. Sampling to ~{target_total} total articles ({target_per_class} per class)...")
        fake_df = fake_df.sample(n=min(target_per_class, len(fake_df)), random_state=42).reset_index(drop=True)
        true_df = true_df.sample(n=min(target_per_class, len(true_df)), random_state=42).reset_index(drop=True)
        print(f"   After sampling:")
        print(f"   Fake: {len(fake_df)} articles")
        print(f"   True: {len(true_df)} articles")
    
    print(f"\n5. Creating train/test split (80/20)...")
    fake_train, fake_test = train_test_split(
        fake_df, test_size=0.2, random_state=42
    )
    true_train, true_test = train_test_split(
        true_df, test_size=0.2, random_state=42
    )
    
    print(f"   Fake - Train: {len(fake_train)}, Test: {len(fake_test)}")
    print(f"   True - Train: {len(true_train)}, Test: {len(true_test)}")
    
    print(f"\n6. Saving split files...")
    fake_train.to_csv(os.path.join(processed_dir, 'FakeTrain.csv'), index=False)
    fake_test.to_csv(os.path.join(processed_dir, 'FakeTest.csv'), index=False)
    true_train.to_csv(os.path.join(processed_dir, 'TrueTrain.csv'), index=False)
    true_test.to_csv(os.path.join(processed_dir, 'TrueTest.csv'), index=False)
    
    print(f"   ✓ Saved to {processed_dir}/")
    print(f"     - FakeTrain.csv ({len(fake_train)} rows)")
    print(f"     - FakeTest.csv ({len(fake_test)} rows)")
    print(f"     - TrueTrain.csv ({len(true_train)} rows)")
    print(f"     - TrueTest.csv ({len(true_test)} rows)")
    
    print(f"\n7. Checking for source bias...")
    fake_texts = fake_train['text'].fillna('').astype(str)
    true_texts = true_train['text'].fillna('').astype(str)
    
    fake_reuters = fake_texts.str.contains('reuters', case=False).sum()
    true_reuters = true_texts.str.contains('reuters', case=False).sum()
    
    print(f"   Reuters mentions:")
    print(f"     Fake: {fake_reuters}/{len(fake_train)} ({fake_reuters/len(fake_train)*100:.1f}%)")
    print(f"     True: {true_reuters}/{len(true_train)} ({true_reuters/len(true_train)*100:.1f}%)")
    
    if true_reuters / len(true_train) > 0.5:
        print(f"   ⚠ WARNING: Still has source bias (Reuters in >50% of true news)")
    else:
        print(f"   ✓ Good: No strong source bias detected")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Review the processed data in processed/ folder")
    print("2. If needed, run clean_data.py to remove source markers")
    print("3. Generate vocab: cd src/data && python3 generate_dict.py")
    print("4. Generate IDs: cd src/data && python3 create_vector_dataset.py")
    print("=" * 60)

if __name__ == '__main__':
    process_welfake()

