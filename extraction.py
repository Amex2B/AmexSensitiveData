from datasets import load_dataset
import pandas as pd

# Load the dataset
ds = load_dataset("Isotonic/pii-masking-200k")

def extract_tel_data(dataset):
    tel_data = []

    # Iterate over each example in the dataset
    for example in dataset:
        privacy_mask = eval(example['privacy_mask'])  # Convert string to dictionary
        
        # Check if '[TEL]' exists in the privacy_mask
        if any('[PHONENUMBER_1]' in key for key in privacy_mask.keys()):
            tel_data.append({
                'unmasked_text': example['unmasked_text'],
                'masked_text': example['masked_text']
            })

    return tel_data

# Extract data from the train set
tel_data = extract_tel_data(ds['train'])

# Convert extracted data to a DataFrame for saving
df = pd.DataFrame(tel_data)

# Save the DataFrame to a CSV file
output_csv = 'extracted_tel_data.csv'
df.to_csv(output_csv, index=False)

print(f"Extracted data saved to {output_csv}")
