import pandas as pd
import torch
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments

# Load your dataset
df = pd.read_csv('maker_day_shrieyaa_stella_mini200_df.csv')

# Print columns to check their names
print(df.columns)

# Preprocess your data into the required format
def preprocess_data(df):
    data = {'tokens': [], 'ner_tags': []}
    
    for _, row in df.iterrows():
        # Use masked_text as the input text
        text = row['masked_text']
        # Split text into tokens
        tokens = text.split()
        
        # Create labels based on masked placeholders
        ner_tags = []
        for token in tokens:
            if token.startswith("[") and token.endswith("]"):
                ner_tags.append(1)  # 1 for sensitive information
            else:
                ner_tags.append(0)  # 0 for non-sensitive information
        
        data['tokens'].append(tokens)
        data['ner_tags'].append(ner_tags)

    return data

# Convert DataFrame to Dataset
data = preprocess_data(df)
dataset = Dataset.from_dict(data)

# Load the tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs.input_ids[i])  # Create a -100 label for padding
        for j, label in enumerate(label):
            if word_ids[j] is not None:  # Ignore padding
                label_ids[word_ids[j]] = label
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Prepare the dataset for training
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # You can use a separate validation set
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("distilbert-ner-model")

device = torch.device("cpu")  # or "mps" if you want to use MPS
model.to(device)

def redact_text(text):
    model.eval()
    tokens = tokenizer(text.split(), truncation=True, padding='max_length', return_tensors='pt').to(device)  # Move to device

    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=2)

    redacted_text = []
    for token, pred in zip(tokens['input_ids'][0], predictions[0]):
        label = pred.item()
        if label != -100:  # Skip padding
            if label == 1:  # Assuming 1 is the label for sensitive information
                redacted_text.append("[REDACTED]")
            else:
                redacted_text.append(tokenizer.decode([token], skip_special_tokens=True))

    return " ".join(redacted_text)

sample_text = "Hello Mr. John Doe, your meeting is scheduled at 123 Main Street, New York on 01/01/2024."
redacted_output = redact_text(sample_text)
print(redacted_output)

sample_text2 = "Please contact Jane Doe at janedoe@example.com for more details."
redacted_output2 = redact_text(sample_text2)
print(redacted_output2)
