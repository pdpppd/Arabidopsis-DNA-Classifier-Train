import pandas as pd

# Step 1: Read the CSV file
file_path = 'arab_ranges_data.csv'  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Step 2: Filter the DataFrame to keep only the specified columns
columns_to_keep = ['seqname', 'feature', 'start', 'end']
filtered_df = df[columns_to_keep]

# Step 3: Save the filtered DataFrame to a new CSV file
output_file_path = 'filtered_arab_info.csv'  # Change this to your desired output file name
filtered_df.to_csv(output_file_path, index=False)  # `index=False` so that Pandas doesn't write row indices to the CSV file

print(f"Filtered CSV has been saved as '{output_file_path}'.")
