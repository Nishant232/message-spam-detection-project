# Spam Detection Project

This project is a comprehensive implementation of spam detection using multiple machine learning models and fine-tuned transformer models. The objective is to classify text messages as either "Spam" or "Not Spam."

## Project Overview
The notebook demonstrates the following:
1. Loading and preprocessing textual data from an Excel file.
2. Implementing a Naive Bayes classifier with TF-IDF vectorization.
3. Utilizing a pre-trained Hugging Face DistilBERT model for sentiment analysis.
4. Fine-tuning the DistilBERT model for the spam classification task.
5. Comparing performance between traditional machine learning and transformer-based approaches.

## Prerequisites
Before running the notebook, ensure you have the required libraries installed and your data formatted correctly.

### Required Libraries
- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning model training and evaluation.
- **transformers**: For using pre-trained transformer models and fine-tuning.
- **tensorflow**: For building, training, and saving fine-tuned models.
- **openpyxl**: For reading Excel files.

### Dataset
The dataset must be an Excel file named `data.xlsx` with the following structure:
- **First Column**: Labels (e.g., `Spam`, `Not Spam`).
- **Second Column**: Text messages to classify.

## Steps in the Notebook

### 1. Data Loading and Preprocessing
- The dataset is loaded using pandas and preprocessed to lower-case all text.
- Data is split into training and test sets using `train_test_split` from scikit-learn.

### 2. Naive Bayes Classifier
- **TF-IDF Vectorization**: Text features are extracted using the `TfidfVectorizer` with stop words removed.
- **Model Training**: A Multinomial Naive Bayes classifier is trained on the TF-IDF vectors.
- **Performance Evaluation**: Accuracy and classification reports are generated.

### 3. Pre-trained Hugging Face Model
- A pre-trained DistilBERT model is loaded for sentiment analysis.
- Sentiment predictions are made directly on input messages without fine-tuning.

### 4. Fine-tuning DistilBERT
- **Tokenizer**: The text data is tokenized using the DistilBERT tokenizer with truncation and padding.
- **Dataset Preparation**: The tokenized data is converted into TensorFlow datasets for training and evaluation.
- **Fine-tuning**: The pre-trained DistilBERT model is fine-tuned for binary classification.
- **Evaluation**: Performance of the fine-tuned model is assessed on the test set.

### 5. Model Saving and Loading
- The fine-tuned model is saved to a directory and can be reloaded for future predictions.
- Sample messages are tested with the loaded model to demonstrate its usage.

## How to Run the Notebook
1. Install the required libraries using pip:
   ```bash
   pip install pandas scikit-learn transformers openpyxl tensorflow
   ```
2. Place your dataset (`data.xlsx`) in the same directory as the notebook.
3. Run the cells sequentially to execute the workflow.

## Example Usage
Below is an example of how the models classify a given message:

- **Input Message**: "You've won a $1000 prize!"
- **Naive Bayes Prediction**: Spam
- **Hugging Face Model Prediction**: Spam with confidence 0.53

## Future Work
- Explore other transformer models (e.g., BERT, RoBERTa).
- Implement hyperparameter tuning for better performance.
- Extend the dataset for multilingual spam detection.

## Author
Nishant

[![GitHub](https://img.shields.io/badge/GitHub/Nishant232-181717?style=flat&logo=github&logoColor=white)](https://github.com/Nishant232)

[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn/2005Nishant-0A66C2?logo=linkedin-white&logoColor=fff)](www.linkedin.com/in/2005nishant)
