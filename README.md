# Sensitive Data Discovery with American Express
**PII Redaction Text Classifier**  

This project builds a machine learning model to detect and redact sensitive personal information (PII) from internal text data, enhancing customer privacy and generating compliant test datasets.

### 📌 Key Features
- **PII Detection and Redaction**: Automatically detects and redacts sensitive data types, including names, addresses, and financial information.
- **Compliance-Friendly Dataset Generation**: Ensures that internal datasets meet privacy and regulatory requirements by masking PII fields.
- **Scalable and Extensible**: Modular code allows for scaling and adapting to additional PII types or sources.

### 🔧 Techniques Used
- **Preprocessing and Vectorization**: Utilizes scikit-learn for efficient data processing and transformation.
- **Deep Learning with DeBERTa**: Employs DeBERTa for high-accuracy PII entity recognition and context-aware redaction.
- **Regex-Based Masking**: Complements DeBERTa with Regex for faster, pattern-based PII detection, enhancing model accuracy for well-defined data patterns.

### 📚 Dataset
[PII Masking Dataset (200k)](https://huggingface.co/datasets/Isotonic/pii-masking-200k): A dataset with labeled PII used to train and validate the redaction model.

### Contributors:
- Ayan Gaur
- Joy Chang
- Shrieyaa Sekar Jayanthi
- Shubangi Waldiya
- Stella Huang
- Break Through Tech AI at UCLA & American Express
