# Spam Classifier using BERT

This project is a spam classifier built using BERT (Bidirectional Encoder Representations from Transformers) model. The classifier is trained to distinguish between spam and ham (non-spam) messages.

## Dataset

The dataset used in this project is the `spam.csv` file. It contains two columns:
- `Category`: Indicates whether the message is spam or ham.
- `Message`: The text of the message.

## Model Architecture

The model is built using the `TFBertModel` from the Hugging Face Transformers library. The architecture includes a BERT model followed by a dense layer with a sigmoid activation function.

## Performance

The model achieved an accuracy of 51% on the test dataset.

## Setup and Installation

1. Clone the repository.
2. Install the required packages:
    ```bash
    pip install numpy pandas tensorflow transformers scikit-learn
    ```

3. Ensure you have a GPU available for faster training and inference.

## Training the Model

The model can be trained using the provided script. The steps include:
1. Loading and preprocessing the dataset.
2. Tokenizing the text data using BERT tokenizer.
3. Resampling the dataset to handle class imbalance.
4. Splitting the data into training and testing sets.
5. Defining and compiling the model.
6. Training the model.
7. Saving the model architecture and weights.

### Training Script


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import TFAutoModel, AutoTokenizer, TFBertModel
from sklearn.utils import resample

# Check for GPU
device = tf.config.experimental.list_physical_devices('GPU')
if device:
    tf.config.experimental.set_memory_growth(device[0], True)

# Load and preprocess dataset
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
data_set = pd.read_csv('spam.csv')
data_set['spam'] = data_set['Category'].apply(lambda x: 1 if x == 'spam' else 0)

spam = data_set[data_set['Category'] == 'spam']
ham = data_set[data_set['Category'] == 'ham']

# Resample to handle class imbalance
resampled_ham = resample(ham, replace=True, n_samples=len(spam), random_state=42)
resampled_data = pd.concat([resampled_ham, spam])

# Prepare data for training and testing
text = resampled_data['Message'].tolist()
labels = resampled_data['spam'].values
X_train, X_test, y_train, y_test = train_test_split(text, labels, stratify=labels, test_size=0.2, random_state=42)

train_output = tokenizer(X_train, padding=True, truncation=True, max_length=128, return_tensors='tf')
test_output = tokenizer(X_test, padding=True, truncation=True, max_length=128, return_tensors='tf')

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_output), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_output), y_test))

batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Define the model
class SpamClassifier(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SpamClassifier, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        bert_output = self.bert(inputs)[1]
        return self.dense(bert_output)

# Instantiate and compile the model
model = SpamClassifier()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model architecture as a JSON string
model_json = model.to_json()
with open("Sentiment_analysis.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("Sentiment_analysis_weights.h5")

# Evaluate the model
loss, accuracy = model.evaluate(dict(test_output), y_test, verbose=2)
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Overall Loss: {loss:.2f}')

# Save the model architecture and weights again
model_json = model.to_json()
with open("Sentiment_analysis.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Sentiment_analysis_weights.h5")
model.summary()
