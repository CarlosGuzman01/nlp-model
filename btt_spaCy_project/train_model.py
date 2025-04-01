# Import necessary libraries
import spacy
from spacy.util import minibatch
from spacy.training import Example
import random
import pandas as pd

import time

start_time = time.time()  # Start time
# Your code here

# Create a blank English NLP pipeline
nlp = spacy.blank("en")

print("started training my model...")

# Add a text categorization component
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat")
else:
    textcat = nlp.get_pipe("textcat")

# Define labels for classification
labels = ["FOOD", "SPORTS", "BOOKS"]
for label in labels:
    textcat.add_label(label)

# Load training data
df = pd.read_csv("train.csv")

# Convert data into Example objects
train_data = []
for _, row in df.iterrows():
    doc = nlp.make_doc(row["text"])
    cats = {label: label == row["label"] for label in labels}
    example = Example.from_dict(doc, {"cats": cats})
    train_data.append(example)

# Train the model
optimizer = nlp.initialize()
for epoch in range(20):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=2)
    for batch in batches:
        nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch+1} Loss: {losses}")

# Save the model
nlp.to_disk("text_classifier_model_20")
print("✅ Model trained and saved to 'text_classifier_model'")


end_time = time.time()  # End time

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
