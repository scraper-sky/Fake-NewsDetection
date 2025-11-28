import pandas as pd
import json
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_new_data():
    print("=" * 60)
    print("SETTING UP NEW DATASET FOR TRAINING")
    print("=" * 60)
    
    new_data_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(new_data_dir, 'processed')
    main_data_dir = os.path.join(new_data_dir, '..')
    id_data_dir = os.path.join(main_data_dir, 'ID-data')
    
    print("\n1. Cleaning processed files...")
    files = ['FakeTrain.csv', 'FakeTest.csv', 'TrueTrain.csv', 'TrueTest.csv']
    for file in files:
        filepath = os.path.join(processed_dir, file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if 'title' in df.columns and 'text' in df.columns:
                df_clean = df[['title', 'text']].copy()
                df_clean.to_csv(filepath, index=False)
                print(f"   ✓ Cleaned {file}: {len(df_clean)} rows")
    
    print("\n2. Copying to main data directory...")
    for file in files:
        src = os.path.join(processed_dir, file)
        dst = os.path.join(main_data_dir, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"   ✓ Copied {file}")
    
    print("\n3. Creating TrainingData.csv...")
    fake_train = pd.read_csv(os.path.join(main_data_dir, 'FakeTrain.csv'))
    true_train = pd.read_csv(os.path.join(main_data_dir, 'TrueTrain.csv'))
    training_data = pd.concat([fake_train, true_train], ignore_index=True)
    training_data.to_csv(os.path.join(main_data_dir, 'TrainingData.csv'), index=False)
    print(f"   ✓ Created TrainingData.csv ({len(training_data)} rows)")
    
    print("\n4. Generating vocabulary...")
    os.chdir(main_data_dir)
    from generate_dict import get_dict
    vocab_dict = get_dict()
    with open('vocab_dict.json', 'w') as f:
        json.dump(vocab_dict, f, indent=4)
    print(f"   ✓ Generated vocab_dict.json ({len(vocab_dict)} words)")
    
    print("\n5. Generating subject dictionary...")
    subject_dict = {'<PAD>': 0, '<UNK>': 1}
    with open('subject_dict.json', 'w') as f:
        json.dump(subject_dict, f, indent=4)
    print("   ✓ Generated subject_dict.json (placeholder)")
    
    print("\n6. Generating ID files...")
    os.makedirs(id_data_dir, exist_ok=True)
    os.chdir(id_data_dir)
    
    from create_vector_dataset import get_maximum, create_json, get_id
    import json as json_module
    
    with open(os.path.join(main_data_dir, 'vocab_dict.json'), 'r') as f:
        vocab_dict = json_module.load(f)
    with open(os.path.join(main_data_dir, 'subject_dict.json'), 'r') as f:
        subject_dict = json_module.load(f)
    
    title_len, text_len = get_maximum(os.path.join(main_data_dir, 'TrainingData.csv'))
    print(f"   Max lengths: title={title_len}, text={text_len}")
    
    create_json(
        os.path.join(main_data_dir, 'FakeTrain.csv'),
        'FakeIDTrain.csv',
        vocab_dict, subject_dict, title_len, text_len, 0
    )
    create_json(
        os.path.join(main_data_dir, 'FakeTest.csv'),
        'FakeIDTest.csv',
        vocab_dict, subject_dict, title_len, text_len, 0
    )
    create_json(
        os.path.join(main_data_dir, 'TrueTrain.csv'),
        'TrueIDTrain.csv',
        vocab_dict, subject_dict, title_len, text_len, 1
    )
    create_json(
        os.path.join(main_data_dir, 'TrueTest.csv'),
        'TrueIDTest.csv',
        vocab_dict, subject_dict, title_len, text_len, 1
    )
    print("   ✓ Generated ID files")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("New dataset is ready for training.")
    print("Run: python -m src.models.train")
    print("=" * 60)

if __name__ == '__main__':
    setup_new_data()

