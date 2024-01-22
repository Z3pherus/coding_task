import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Configuration
batch_size = 64
learning_rate = 2e-5
model_path = "models/fine_tuned_bert.pt"
n_epochs = 5
random_state = 42
tag_threshold = 100

# Set the logging level
logging.getLogger().setLevel(logging.INFO)

# Load the data
df_tags = pd.read_csv("data/tags.csv", encoding="ISO-8859-1")
df_quest = pd.read_csv("data/questions.csv", encoding="ISO-8859-1", usecols=["Id", "Title", "Body"])
logging.info("Data loaded.")

n_tags_before = df_tags["Tag"].nunique()

# Count the occurrences of each tag
tag_counts = df_tags["Tag"].value_counts()

# Identify tags that occur at least 100 times
frequent_tags = tag_counts[tag_counts >= tag_threshold].index

# All tags with occurrences of less than 100 are removed
# This ensures that for each tag there are enough examples in the three datasets (train, val and test)
df_tags = df_tags[df_tags["Tag"].isin(frequent_tags)]

n_tags_after = df_tags["Tag"].nunique()
n_tags_removed = n_tags_before - n_tags_after
logging.info(f"{n_tags_removed} tags were removed due to low occurrences! {n_tags_after} tags remaining.")

# Now we have to remove all questions that do not have any tag anymore
# Apparently there are no such questions, so nothing gets removed
n_rows_before = len(df_quest.index)

df_quest = df_quest[df_quest["Id"].isin(df_tags["Id"])]

n_rows_after = len(df_quest.index)
n_rows_removed = n_rows_before - n_rows_after
if n_rows_removed > 0:
    logging.info(f"{n_rows_removed} rows were removed due to missing tags! {n_rows_after} rows remaining.")


# A very simple preprocessing is performed
# This can certainly be improved
def preprocess_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Replace \n and \r with a whitespace
    text = text.replace("\n", " ").replace("\r", " ")

    # Decode &lt; and &gt;
    text = text.replace("&lt;", "<").replace("&gt;", ">")

    # Remove multiple whitespaces with single whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase text
    text = text.lower()

    return text


# Apply the preprocessing to the two text columns
df_quest["Title"] = df_quest["Title"].apply(preprocess_text)
df_quest["Body"] = df_quest["Body"].apply(preprocess_text)

# Combine the two columns Title and Body to one column
df_quest["Text"] = df_quest["Title"] + " " + df_quest["Body"]

# Combine the tags into one row
df_tags = df_tags.groupby("Id")["Tag"].apply(list).reset_index()

# Ont-Hot-Encode tags
mlb = MultiLabelBinarizer()
df_tags["Tag"] = mlb.fit_transform(df_tags["Tag"]).tolist()

logging.info("Columns processed.")

# Get tokenizer for BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_text(text):
    # Bert supports a maximal number of 512 tokens
    return tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


# Tokenize text
df_quest["Text_token"] = df_quest["Text"].apply(tokenize_text)

logging.info("Text tokenized.")

# Drop unneeded columns
df_quest = df_quest.drop(["Title", "Body", "Text"], axis=1)

# Split the data
train_ids, temp_df = train_test_split(df_quest[["Id"]], test_size=0.2, random_state=random_state)
val_ids, test_ids = train_test_split(temp_df, test_size=0.5, random_state=random_state)

logging.info("Performed train-validation-test split.")


class QuestionDataset(Dataset):
    def __init__(self, ids, data_quest, data_tags):
        # Filter for IDs of the corresponding split
        data_quest_filt = data_quest[data_quest["Id"].isin(ids["Id"])]
        data_tags_filt = data_tags[data_tags["Id"].isin(ids["Id"])]

        self.labels = data_tags_filt["Tag"].to_list()
        self.texts = data_quest_filt["Text_token"].to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get input_ids and attention_mask from pre-tokenized text
        input_ids = self.texts[idx]["input_ids"].squeeze(0)
        attention_mask = self.texts[idx]["attention_mask"].squeeze(0)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)

        return input_ids, attention_mask, labels


# Create datasets
train_dataset = QuestionDataset(train_ids, df_quest, df_tags)
val_dataset = QuestionDataset(val_ids, df_quest, df_tags)
test_dataset = QuestionDataset(test_ids, df_quest, df_tags)

del df_quest, df_tags, train_ids, val_ids, test_ids

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

logging.info("Created datasets and data loaders.")


# A pretrained BERT for the multi-label classification is used
# Only the output layer is replaced
class BertForMultiLabelClassification(torch.nn.Module):
    def __init__(self, n_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(bert_outputs[1])


# Instantiate the model
n_classes = len(mlb.classes_)
model = BertForMultiLabelClassification(n_classes)

# Freeze all layers in BERT model
for param in model.bert.parameters():
    param.requires_grad = False

# Enable training for the classifier layer
# Since the dataset is huge, only the classifier part will be trained to speed up the training
for param in model.classifier.parameters():
    param.requires_grad = True

logging.info("Model created.")

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device used: {device}")

# Move model to device
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

# Calculate number of batches in training dataset
n_train = len(train_dataset)
n_batches = np.ceil(n_train / batch_size)
logging.info(f"Number of batches with a size of {batch_size}: {n_batches}")

# Perform training
best_loss = 9999.
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for b, batch in enumerate(train_loader, start=1):
        # Get data for this batch
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        # Forward pass
        b_outputs = model(b_input_ids, b_attention_mask)
        loss = criterion(b_outputs, b_labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print current train loss
        if b % 50 == 0:
            curr_loss = total_loss / b
            logging.info(f"Average training loss after batch {b}/{n_batches}: {curr_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    val_labels = []
    val_scores = []
    val_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            # Get data for this batch
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Get predictions of model
            b_outputs = model(b_input_ids, b_attention_mask)
            loss = criterion(b_outputs, b_labels)
            total_val_loss += loss.item()

            b_scores = torch.sigmoid(b_outputs).cpu().numpy()
            b_predictions = b_scores > 0.5
            b_labels = b_labels.cpu().numpy()

            # Save predictions and true labels
            val_labels.append(b_labels)
            val_scores.append(b_scores)
            val_predictions.append(b_predictions)

    # Concatenate all batches
    val_labels = np.vstack(val_labels)
    val_scores = np.vstack(val_scores)
    val_predictions = np.vstack(val_predictions)

    # Calculate average loss and metrics
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_predictions, average="micro")
    val_roc = roc_auc_score(val_labels, val_scores, average="micro")
    val_acc = accuracy_score(val_labels, val_predictions)

    # Check if the current model is better than the previous model
    if avg_val_loss < best_loss:
        torch.save(model.state_dict(), model_path)
        logging.info(f"Epoch {epoch + 1}: New best model saved to {model_path}")

    # Print metrics
    logging.info(f"Epoch {epoch+1} - train: loss = {avg_train_loss:.4f}")
    logging.info(f"Epoch {epoch+1} - validation: loss = {avg_val_loss:.4f}, f1 = {val_f1:.4f}, roc = {val_roc:.4f}, "
                 f"acc = {val_acc:.4f}")

# Load best model
model.load_state_dict(torch.load(model_path))

# Test
model.eval()
total_test_loss = 0
test_labels = []
test_scores = []
test_predictions = []
with torch.no_grad():
    for batch in test_loader:
        # Get data for this batch
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Get predictions of model
        b_outputs = model(b_input_ids, b_attention_mask)
        loss = criterion(b_outputs, b_labels)
        total_test_loss += loss.item()

        b_scores = torch.sigmoid(b_outputs).cpu().numpy()
        b_predictions = b_scores > 0.5
        b_labels = b_labels.cpu().numpy()

        # Save predictions and true labels
        test_labels.append(b_labels)
        test_scores.append(b_scores)
        test_predictions.append(b_predictions)

# Concatenate all batches
test_labels = np.vstack(test_labels)
test_scores = np.vstack(test_scores)
test_predictions = np.vstack(test_predictions)

# Calculate average loss and metrics
avg_test_loss = total_test_loss / len(test_loader)
test_f1 = f1_score(test_labels, test_predictions, average="micro")
test_roc = roc_auc_score(test_labels, test_scores, average="micro")
test_acc = accuracy_score(test_labels, test_predictions)

logging.info(f"Results for the best model on the test data: loss = {avg_test_loss:.4f}, f1 = {test_f1:.4f}, "
             f"roc = {test_roc:.4f}, acc = {test_acc:.4f}")
