# Prediction of tags for questions on stackoverflow

## Prerequisites
- Python 3.10
- Python packages listed in requirements.txt

## Installation
Clone the repository:
```bash
git clone https://github.com/Z3pherus/coding_task
cd coding_task
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Files
- **eda.ipynb**: Jupyter notebook that performs a detailed explanatory data analysis of the tags.csv
and questions.csv files.
- **train_model.py**: Python file that uses the packages PyTorch and transformers to train a BERT model on the data.

## Dataset
- The dataset consists of questions from stackoverflow posted between 2008 and 2016
- Each question is associated with a title
- There are more than 600k questions and more than 16k tags
- Most tags are rare and occur sometimes only once
- Most questions are related to python
- Each question has between one and five tags (three on average)
- Each question has at least one common tag (> 100 occurrences)

## Multi-label classification model
- The Task is to predict the tags of stackoverflow questions, which is a multi-label classification problem
- BERT is used as model since it is one of the most popular transformer models for classification
- Rare tags with occurrences of less than 100 are ignored to have enough examples for each tag in the validation 
and test datasets
- The data is randomly split into train (80%), validation (10%) and test (10%) datasets
- A very simple preprocessing is used for the title and the question, which for example removes HTML tags and newline
and carriage return characters
- To keep the training time reasonable (the dataset is huge), a pretrained BERT model is used and only the classification
layer is trained
- The model with the lowest validation loss is stored and tested on the test data

## Ideas to improve the model
- Full fine-tuning or full training of the BERT model
- Improved preprocessing of the Title and Body columns (e.g. handling of special characters)
- Test of other models (e.g. BigBird) since BERT is limited to 512 tokens and some questions are longer
- (If possible) better sampling of the training data so that for all relevant tags there are enough examples
- Stratification of the train-val-test split (so that the distributions of the tags are the same in all three datasets)
  (complicated for such a huge multi-label classification task)
- Hyperparameter tuning (e.g. batch size, learning rate)
