import pandas as pd
from Bio import SeqIO
from itertools import product
from tqdm import tqdm
import numpy as np

def generate_kmer_vector(sequence, k=6):
    # Initialize an empty dictionary for k-mer counts
    kmer_counts = {"".join(kmer): 0 for kmer in product('ACGT', repeat=k)}
    
    # Count k-mers in the sequence
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
    
    # Convert the counts to a vector and apply softmax normalization
    kmer_vector = np.array(list(kmer_counts.values()))
    kmer_vector = np.exp(kmer_vector) / np.sum(np.exp(kmer_vector))
    
    return kmer_vector.tolist()

def process_data(csv_file, fasta_file, output_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Read the fasta file and store sequences in a dictionary
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    
    # Prepare an empty list to store the output data
    output_data = []
    
    # Process each row in the dataframe with a tqdm progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        seq_id = row['seqname']
        start, end = row['start'], row['end']
        feature = row['feature']
        
        # Extract the relevant sequence
        if seq_id in sequences:
            sequence = sequences[seq_id][start-1:end]  # -1 because Python indexing starts at 0
            kmer_vector = generate_kmer_vector(sequence)
            output_data.append({'Vector': kmer_vector, 'Feature': feature})
    
    # Convert the output data to a DataFrame and save it as a CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Processed data has been saved to {output_file}")

if __name__ == "__main__":
    csv_file = 'balanced_arab_data.csv'  # Change to your CSV file path
    fasta_file = 'arabidopsis_genome.fasta'  # Change to your fasta file path
    output_file = 'processed_kmer_vectors_softmax.csv'  # Desired output file name
    process_data(csv_file, fasta_file, output_file)