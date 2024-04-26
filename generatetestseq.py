from Bio import SeqIO
import pandas as pd

def load_genome(fasta_path):
    """Load the genome FASTA file and adjust keys to use only the first part of the header."""
    genome = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header_parts = record.id.split()  # Assuming the header is split by spaces
        key = f">{header_parts[0]}"  # Create a dictionary key from the first part of the header
        genome[key] = record
    return genome

def extract_sequences(data, genome, feature_type, num_sequences=5):
    """Extract and print sequences for a specific feature type."""
    feature_data = data[data['feature'] == feature_type]
    for i, row in feature_data.head(num_sequences).iterrows():
        seq_id = f">{row['seqname']}"
        try:
            sequence = genome[seq_id].seq[row['start']-1:row['end']]
            print(f"{seq_id} | {feature_type} | Start: {row['start']} | End: {row['end']}")
            print(sequence)
            print("-" * 80)
        except KeyError:
            print(f"KeyError: Sequence ID {seq_id} not found in the genome.")

def main():
    # Path to the genome FASTA file
    genome_path = "arabidopsis_genome.fasta"
    # Load the genome
    genome = load_genome(genome_path)
    
    # Load your data
    data_path = "balanced_arab_data.csv"
    data = pd.read_csv(data_path)
    
    # Define the features to extract
    features = ['CDS', 'gene', 'transposon_fragment', 'mRNA']
    
    # Extract and print sequences for each feature type
    for feature in features:
        extract_sequences(data, genome, feature)

if __name__ == "__main__":
    main()
