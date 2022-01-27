# Slightly modified version of "finetune_models.py" that handles CV in a multilingual setting.
# In order to achieve a balanced split between languages. each fold is stratified based on language code.

import os
import argparse
import glob
import logging
import torch
import pandas as pd
import numpy as np
from shutil import rmtree
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

logging = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

seed = 1337

parser = argparse.ArgumentParser(description='Select language df')
parser.add_argument('--train_data', type=str, default=None,
                    help='name of csv file with train data')
parser.add_argument('--test_data', type=str, default=None,
                    help='name of csv file with test data')
parser.add_argument('--output_name', type=str,
                    help="name of directory to save the model")
parser.add_argument('--output_identifier', type=str,
                    help="identifier for multiple cross validation runs results")
parser.add_argument("--model", type=str, help=f'use any of:'
                    f' # cardiffnlp/twitter-xlm-roberta-base-sentiment'
                    f' # cardiffnlp/twitter-xlm-roberta-base'
                    f' # cardiffnlp/bertweet-base-sentiment  (ENGLISH ONLY)'
                    f' # cardiffnlp/twitter-roberta-base-sentiment (ENGLISH ONLY)'
                    f' # roberta-base (ENGLISH ONLY)'
                    f' # xlm-roberta-base'
                    f' # BSC-TeMU/roberta-base-bne (SPANISH ONLY)'
                    f' # gealexandri/palobert-base-greek-uncased-v1 (GREEK ONLY)'
                    )


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


args = parser.parse_args()
logging.info(args)

model_name = args.model


df = pd.read_csv(args.train_data)
# map to correct labels
df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})
df['label'] = df['label'].astype(int)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
skf.get_n_splits(df.index)

cv_results = {'uk': [], 'es': [], 'gr': []}

# stratify based on language code
for train_val_index, test_index in skf.split(df.index, df['code'].tolist()):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3)
    # must used different tokenizer in case of  cardiffnlp/bertweet-base-sentiment or cardiffnlp/twitter-roberta-base-sentiment
    if 'twitter-xlm-roberta-base-sentiment' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            'cardiffnlp/twitter-xlm-roberta-base-sentiment', max_len=512, normalization=True)
    elif 'twitter-xlm-roberta-base' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            'cardiffnlp/twitter-xlm-roberta-base', max_len=512, normalization=True)
    elif 'xlm-roberta-base' in model_name:
        logging.info('Using xlm-roberta-base.')
        tokenizer = AutoTokenizer.from_pretrained(
            'xlm-roberta-base', max_len=512, normalization=True)

    df_train_val = df.loc[train_val_index]
    df_test = df.loc[test_index]

    # split to train/validation sets
    train_idx, validation_idx = train_test_split(
        df_train_val.index, test_size=0.15)
    df_train = df_train_val.loc[train_idx]
    train_encodings = tokenizer(
        df_train['text'].tolist(),  padding=True, truncation=True)
    train_dataset = TweetDataset(train_encodings, df_train['label'].tolist())

    df_validation = df_train_val.loc[validation_idx]
    validation_encodings = tokenizer(
        df_validation['text'].tolist(), padding=True, truncation=True)
    validation_dataset = TweetDataset(
        validation_encodings, df_validation['label'].tolist())

    callbacks = EarlyStoppingCallback()
    training_args = TrainingArguments(
        output_dir=f"{args.output_name}",
        save_strategy='epoch',
        overwrite_output_dir=False,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        learning_rate=5e-5,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        logging_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[EarlyStoppingCallback(3, 0.01)]
    )

    trainer.train()

    # make predictions
    test_encodings = tokenizer(
        df_test['text'].tolist(),  padding=True, truncation=True)
    test_dataset = TweetDataset(test_encodings, df_test['label'].tolist())

    results = trainer.predict(test_dataset)

    # get metrics
    predictions = results[0]
    predictions = np.argmax(predictions, axis=1)
    labels = results.label_ids

    df_test['predictions'] = predictions

    results = {}
    for code in ['uk', 'es', 'gr']:
        df_to_evaluate = df_test[df_test['code'] == code]

        metrics = classification_report(df_to_evaluate['label'].tolist(
        ), df_to_evaluate['predictions'].tolist(), output_dict=True)
        cv_results[code].append(metrics)


# dump the results from all folds
dump(cv_results,
     f"{args.output_name}/metrics_run_{args.output_identifier}.pkl")

# get avg of folds for each lang
for code in ['uk', 'es', 'gr']:
    lang_results = cv_results[code]

    df_results = pd.DataFrame(lang_results[0])
    for fold in lang_results[1:]:
        df_results = df_results.add(pd.DataFrame(fold))

    df_results = df_results / 5
    df_results = df_results.transpose()
    df_results = round(df_results, 2)
    df_results.to_csv(
        f"{args.output_name}/{code}_avg_metrics_run_{args.output_identifier}.csv")


# remove auto-saved runs directories
rmtree(f"{args.output_name}/runs")

# remove checkpoints
checkpoints = glob.glob(f"{args.output_name}/checkpoint*")
for entry in checkpoints:
    rmtree(entry)
