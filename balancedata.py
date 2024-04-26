import pandas as pd

def create_balanced_dataset(input_file, output_file):
    # Load the original dataset
    data = pd.read_csv(input_file)

    # Find the top 8 most abundant features
    top_features = data['feature'].value_counts().head(8).index

    # Initialize an empty DataFrame for the new balanced dataset
    balanced_data = pd.DataFrame()

    # Loop through each of the top 8 features
    for feature in top_features:
        # Filter the data for the current feature
        feature_data = data[data['feature'] == feature]
        
        # Sample 10,000 instances of the current feature, if available
        # If not, take all instances
        sampled_data = feature_data.sample(n=min(10000, len(feature_data)), replace=False, random_state=1)
        
        # Append the sampled data to the balanced dataset
        balanced_data = pd.concat([balanced_data, sampled_data], ignore_index=True)
    
    # Save the new balanced dataset to a new CSV file
    balanced_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file_path = 'filtered_arab_info.csv'  # Update this path
    output_file_path = 'balanced_arab_data.csv'  # Update this path
    create_balanced_dataset(input_file_path, output_file_path)
    print(f'Balanced dataset saved to {output_file_path}')
