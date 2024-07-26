#  transformers datasets torch pandas numpy scikit-learn matplotlib seaborn optuna

used for loading dataset

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install  transformers datasets torch pandas numpy scikit-learn matplotlib seaborn optuna.

```bash
pip install transformers datasets torch pandas numpy scikit-learn matplotlib seaborn optuna
```

## importing

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
```

## loading dataset
~~~
dataset = load_dataset('snli')
train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])
test_df = pd.DataFrame(dataset['test'])
~~~

## Data preprocessing  
~~~
train_df.groupby('label').count()
train_df=train_df[train_df['label']!=-1]
train_df.groupby('label').count()
~~~
## training of data
~~~
train_df=train_df[:7500]
train_df
val_df.groupby('label').count()
val_df=val_df[val_df['label']!=-1]
val_df.groupby('label').count()
val_df=val_df[:2500]
~~~
## importing of model
~~~
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(df):
  return tokenizer(df['premise'].tolist(), df['hypothesis'].tolist(), truncation=True, padding='max_length', max_length=128)
train_encodings = tokenize(train_df)
val_encodings = tokenize(val_df)
~~~
## encoding
~~~
class SNLIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SNLIDataset(train_encodings, train_df['label'].values)
val_dataset = SNLIDataset(val_encodings, val_df['label'].values)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
~~~

## validation phase
~~~
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name='linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

best_val_accuracy = 0
early_stopping_patience = 3
patience_counter = 0

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            val_labels.extend(batch['labels'].cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}: Validation Accuracy = {val_accuracy}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
~~~
## Evaluation
~~~

def evaluate(loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    return preds, labels
~~~
## Accuracy
~~~
test_preds, test_labels = evaluate(test_loader)
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy}")
~~~
## Classification Report
~~~
print("Classification Report:")
print(classification_report(test_labels, test_preds, target_names=['entailment', 'neutral', 'contradiction']))
~~~
## Confusion matrix
~~~
conf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['entailment', 'neutral', 'contradiction'], yticklabels=['entailment', 'neutral', 'contradiction'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
~~~
## Prediction
~~~
def predict(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()

    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    return label_map[predicted_label]
~~~
## Input
~~~
premise_=input()
hypothesis_=input()
~~~
## Output
~~~
predict(premise_,hypothesis_)
~~~
