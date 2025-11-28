import pandas as pd
import re
import os

def remove_source_markers(text):
    if pd.isna(text):
        return text
    
    text = str(text)
    
    text = re.sub(r'\bReuters\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'REUTERS', '', text)
    
    text = re.sub(r'Photo by [^\.]+\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Image via [^\.]+\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Featured image via [^\.]+\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'via Getty Images', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Getty Images', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'^[A-Z\s]+ \(Reuters\)\s*-?\s*', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'pic\.twitter\.com/\w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_dataset(input_path, output_path):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Original size: {len(df)} rows")
    
    if 'text' in df.columns:
        print("Removing source markers from text...")
        df['text'] = df['text'].apply(remove_source_markers)
    
    if 'title' in df.columns:
        print("Removing source markers from title...")
        df['title'] = df['title'].apply(remove_source_markers)
    
    if 'text' in df.columns:
        before = len(df)
        df = df[df['text'].notna() & (df['text'] != '')]
        print(f"Removed {before - len(df)} empty text entries")
    
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path} ({len(df)} rows)")
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 60)
    print("CLEANING DATASET - REMOVING SOURCE MARKERS")
    print("=" * 60)
    
    if os.path.exists(os.path.join(raw_dir, 'fake.csv')):
        clean_dataset(
            os.path.join(raw_dir, 'fake.csv'),
            os.path.join(processed_dir, 'fake_cleaned.csv')
        )
    
    if os.path.exists(os.path.join(raw_dir, 'true.csv')):
        clean_dataset(
            os.path.join(raw_dir, 'true.csv'),
            os.path.join(processed_dir, 'true_cleaned.csv')
        )

