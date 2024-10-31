from datasets import load_dataset
import pandas as pd

# Load the dataset
ds = load_dataset("Isotonic/pii-masking-200k")

def extract_tel_numbers_english(dataset):
    tel_data = []

    # Iterate over each example in the dataset
    for example in dataset:
        # Check if the language is English (assuming 'en' for English)
        if example['language'] == 'en':
            # Convert the privacy_mask string back into a dictionary
            privacy_mask = eval(example['privacy_mask'])

            # Check if '[PHONENUMBER_1]' exists in the privacy_mask
            tel_entries = {key: value for key, value in privacy_mask.items() if '[PHONENUMBER_1]' in key}

            # If we have any '[PHONENUMBER_1]' entries, extract them
            if tel_entries:
                # Iterate over all PHONENUMBER_1 entries (could be one or more)
                for tel_key, tel_value in tel_entries.items():
                    # Add to our tel_data list, each TEL entry as a new row
                    tel_data.append({
                        'unmasked_text': example['unmasked_text'],
                        'masked_text': example['masked_text'],
                        'phone_number': tel_value,
                        'privacy_mask': example['privacy_mask']
                    })

    return tel_data

# Extract phone numbers from the train set, only for English rows
tel_data = extract_tel_numbers_english(ds['train'])

# Convert extracted data to a DataFrame for saving
df = pd.DataFrame(tel_data)

# Save the DataFrame to a CSV file
output_csv = 'extracted_tel_numbers_english.csv'
df.to_csv(output_csv, index=False)

print(f"Extracted phone numbers from English rows saved to {output_csv}")
