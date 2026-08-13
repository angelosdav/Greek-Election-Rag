import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "clean")
    db_dir = os.path.join(base_dir, "vector_db")
    
    print("Loading Sentence Transformer model...")
    # 'intfloat/multilingual-e5-small' is a fast, excellent multilingual embedding model
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    print(f"Initializing ChromaDB at {db_dir}...")
    client = chromadb.PersistentClient(path=db_dir)
    
    # Try to get or create collection
    collection_name = "election_statements"
    try:
        client.delete_collection(name=collection_name)
        print("Deleted existing collection to start fresh.")
    except Exception:
        pass
        
    collection = client.create_collection(name=collection_name)
    
    batch_size = 256
    
    # Iterate through all CSVs in data/clean
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
            
        party_name = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)
        
        print(f"Processing {party_name}...".encode('utf-8', 'replace').decode('cp1252', 'ignore'))
        df = pd.read_csv(filepath)
        
        if 'text' not in df.columns:
            print(f"Skipping {party_name}, no 'text' column.")
            continue
            
        # Drop rows with NaN texts
        df = df.dropna(subset=['text'])
        texts = df['text'].tolist()
        
        if not texts:
            continue
            
        # Chroma IDs must be strings
        ids = [f"{party_name}_{i}" for i in range(len(texts))]
        metadatas = [{"party": party_name} for _ in range(len(texts))]
        
        # For E5 models, the standard is to prefix documents with "passage: "
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        # We need to process in batches to avoid OOM or slow performance
        for i in range(0, len(texts), batch_size):
            end_idx = i + batch_size
            batch_texts = texts[i:end_idx]
            batch_prefixed = prefixed_texts[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            # Generate embeddings
            embeddings = model.encode(batch_prefixed, normalize_embeddings=True).tolist()
            
            # Add to ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=batch_texts, # Store original text without 'passage:' prefix
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Added {len(batch_texts)} chunks for {party_name}".encode('utf-8', 'replace').decode('cp1252', 'ignore'))
            
    print(f"Successfully built Vector DB at {db_dir}!")
    print(f"Total documents in collection: {collection.count()}")

    # Process parliamentary votes
    votes_file = os.path.join(base_dir, "data", "raw_source_material", "parliament_key_votes.csv")
    if os.path.exists(votes_file):
        print("Processing parliamentary votes...")
        df_votes = pd.read_csv(votes_file)
        
        for party_name, group in df_votes.groupby('Party'):
            texts = group['Content'].tolist()
            if not texts:
                continue
            
            ids = [f"{party_name}_vote_{i}" for i in range(len(texts))]
            metadatas = [{"party": party_name} for _ in range(len(texts))]
            prefixed_texts = [f"passage: {text}" for text in texts]
            
            # Since votes are max 30 per party, no need for batching here
            embeddings = model.encode(prefixed_texts, normalize_embeddings=True).tolist()
            
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(texts)} voting chunks for {party_name}".encode('utf-8', 'replace').decode('cp1252', 'ignore'))
            
        print(f"Updated total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main()
