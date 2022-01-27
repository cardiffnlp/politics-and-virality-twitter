
import os
import glob
import logging
import torch
import argparse
import pandas as pd
import numpy as np
from transformers import EarlyStoppingCallback, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from joblib import dump
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from shutil import rmtree

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

logging = logging.getLogger(__name__)


os.environ["WANDB_DISABLED"] = "true"

# set seed
seed = 1337


parser = argparse.ArgumentParser(description='Select language df')
parser.add_argument('--train_data', type=str, default=None,
                    help='name of csv file with train data')
parser.add_argument('--test_data', type=str, default=None,
                    help='name of csv file with test data')
parser.add_argument('--output_name', type=str,
                    help="name of directory to save the model")
parser.add_argument('--output_identifier', type=str,
                    help="identifier for multiple runs results")
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
parser.add_argument('--test_only', default=False,
                    action='store_true', help='only test the given model')
parser.add_argument('--train_only', default=False,
                    action='store_true', help='only train the given model')
parser.add_argument('--cross_validation', default=False,
                    action='store_true', help='perform cross validation')


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


def train_test(df_train_val, df_test, model, tokenizer, cross_validation):
    """
    Train a model given the given train/test sets.

    Args:
        df_train_val (pd.DataFrame): Dataframe containing "text" and "label" columns to be used for training/validation.
        df_test (pd.DataFrame):Dataframe containing "text" and "label" columns to be used for testing.
        model (transformers.AutoModelForSequenceClassification): Transormers model to be used.
        tokenizer (transformers.AutoTokenizer): Tokenizer to be used along with model.
        cross_validation (boolean): Indicates whether the training is part of cross-validation.

    Returns:
        dict: A dictionary containing the metrics of the run as given by sklearn.classification_report
        list: A list with the predicitons on the provided test set.
    """
    logging.info("Train/test given model.")
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

    overwrite_dir = not cross_validation
    training_args = TrainingArguments(
        output_dir=f"{args.output_name}",
        save_strategy='epoch',
        overwrite_output_dir=overwrite_dir,
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

    if not cross_validation:
        # save model
        trainer.save_model(f"{args.output_name}/")

    # make predictions
    test_encodings = tokenizer(
        df_test['text'].tolist(),  padding=True, truncation=True)
    test_dataset = TweetDataset(test_encodings, df_test['label'].tolist())

    results = trainer.predict(test_dataset)

    # get metrics
    predictions = results[0]
    predictions = np.argmax(predictions, axis=1)
    labels = results.label_ids

    metrics = classification_report(labels, predictions, output_dict=True)

    return metrics, predictions


def train_only(df, model, tokenizer):
    """
    Train model provided on the given train set.

    Args:
        df (pd.DataFrame):Dataframe containing "text" and "label" columns to be used for training/validation.
        model (transformers.AutoModelForSequenceClassification): Transormers model to be used.
        tokenizer (transformers.AutoTokenizer): Tokenizer to be used along with model.

    Returns:
        dict: dummy values
        list: dummy values
    """
    logging.info("Only train given model.")

    train_idx, validation_idx = train_test_split(df.index, test_size=0.15)
    df_train = df.loc[train_idx]
    train_encodings = tokenizer(
        df_train['text'].tolist(),  padding=True, truncation=True)
    train_dataset = TweetDataset(train_encodings, df_train['label'].tolist())

    df_validation = df.loc[validation_idx]
    validation_encodings = tokenizer(
        df_validation['text'].tolist(), padding=True, truncation=True)
    validation_dataset = TweetDataset(
        validation_encodings, df_validation['label'].tolist())

    training_args = TrainingArguments(
        output_dir=f"{args.output_name}",
        save_strategy='epoch',
        overwrite_output_dir=True,
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

    # save model
    trainer.save_model(f"{args.output_name}/")

    # return dummy variables, just for cosistency
    return {'dummy': ['foo', 'bar']}, ['dummy']


def test_only(df_test, model, tokenizer):
    """
    Test model provided on the given test set.

    Args:
        df_test (pd.DataFrame):Dataframe containing "text" and "label" columns to be used for testing.
        model (transformers.AutoModelForSequenceClassification): Transormers model to be used.
        tokenizer (transformers.AutoTokenizer): Tokenizer to be used along with model.

    Returns:
        dict: A dictionary containing the metrics of the run as given by sklearn.classification_report
        list: A list with the predicitons on the provided test set.
    """
    logging.info("Only testing given model.")

    training_args = TrainingArguments(
        output_dir=f"{args.output_name}",          # output directory
        save_strategy='no',
        overwrite_output_dir=False,
        logging_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,                  # training arguments, defined above
    )
    # no train

    # make predictions
    test_encodings = tokenizer(
        df_test['text'].tolist(),  padding=True, truncation=True)
    test_dataset = TweetDataset(test_encodings, df_test['label'].tolist())

    results = trainer.predict(test_dataset)

    # get metrics
    predictions = results[0]
    predictions = np.argmax(predictions, axis=1)
    labels = results.label_ids

    metrics = classification_report(labels, predictions, output_dict=True)

    return metrics, predictions


def get_model_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3)
    if 'twitter-xlm-roberta-base-sentiment' in model_name:
        logging.info('Using cardiffnlp/twitter-xlm-roberta-base-sentiment')
        tokenizer = AutoTokenizer.from_pretrained(
            'cardiffnlp/twitter-xlm-roberta-base-sentiment', max_len=512, normalization=True)
    elif 'twitter-xlm-roberta-base' in model_name:
        logging.info('Using cardiffnlp/twitter-xlm-roberta-base')
        tokenizer = AutoTokenizer.from_pretrained(
            'cardiffnlp/twitter-xlm-roberta-base', max_len=512, normalization=True)
    elif 'twitter-roberta-base-sentiment' in model_name:
        logging.info('Using cardiffnlp/twitter-roberta-base-sentiment')
        tokenizer = AutoTokenizer.from_pretrained(
            'cardiffnlp/twitter-roberta-base-sentiment', max_len=512, normalization=True)
    elif 'bertweet-base-sentiment' in model_name:
        logging.info('Using vinai/bertweet-base tokenizer.')
        tokenizer = AutoTokenizer.from_pretrained(
            'vinai/bertweet-base', max_len=512, normalization=True)
    elif 'xlm-roberta-base' in model_name:
        logging.info('Using xlm-roberta-base.')
        tokenizer = AutoTokenizer.from_pretrained(
            'xlm-roberta-base', max_len=512, normalization=True)
    elif 'palobert-base-greek-uncased-v1' in model_name:
        logging.info('Using gealexandri/palobert-base-greek-uncased-v1')
        tokenizer = AutoTokenizer.from_pretrained(
            'gealexandri/palobert-base-greek-uncased-v1', max_len=512, normalization=True)
    elif 'roberta-base-bne' in model_name:
        logging.info('Using BSC-TeMU/roberta-base-bne')
        tokenizer = AutoTokenizer.from_pretrained(
            'BSC-TeMU/roberta-base-bne', max_len=512, normalization=True)
    elif 'roberta-base' in model_name:
        logging.info('Using roberta-base')
        tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base', max_len=512, normalization=True)

    return model, tokenizer


args = parser.parse_args()
logging.info(args)


model_name = args.model
# for cross validation
if args.cross_validation:
    logging.info('Performing cross-validation (5-fold)')
    df = pd.read_csv(args.train_data)
    # map to correct labels
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})
    df['label'] = df['label'].astype(int)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf.get_n_splits(df.index)

    cv_results = []

    for train_index, test_index in kf.split(df.index):
        if not args.test_only:
            df_train = df.loc[train_index]

        df_test = df.loc[test_index]

        model, tokenizer = get_model_tokenizer(model_name)
        if args.test_only:
            metrics, _ = test_only(df_test, model, tokenizer)
        if args.train_only:
            metrics, predictions = train_only(df_train, model, tokenizer)
        if not args.test_only and not args.train_only:
            metrics, _ = train_test(
                df_train, df_test, model, tokenizer, args.cross_validation)

        cv_results.append(metrics)

    # dump the results from all folds
    dump(cv_results, f"{args.output_name}/metrics.pkl")

    # get avg of folds
    df_results = pd.DataFrame(cv_results[0])
    for fold in cv_results[1:]:
        df_results = df_results.add(pd.DataFrame(fold))

    df_results = df_results / 5
    df_results = df_results.transpose()
    df_results = round(df_results, 2)
    df_results.to_csv(
        f"{args.output_name}/avg_metrics_run_{args.output_identifier}.csv")
# for explictly train/test sets
else:
    logging.info('No cross-validation')
    if not args.test_only:
        df_train = pd.read_csv(args.train_data)
        df_train['label'] = df_train['label'].map({-1: 0, 0: 1, 1: 2})
        df_train['label'] = df_train['label'].astype(int)

    if not args.train_only:
        df_test = pd.read_csv(args.test_data)
        df_test['label'] = df_test['label'].map({-1: 0, 0: 1, 1: 2})
        df_test['label'] = df_test['label'].astype(int)

    model, tokenizer = get_model_tokenizer(model_name)
    if args.test_only:
        metrics, predictions = test_only(df_test, model, tokenizer)
    if args.train_only:
        metrics, predictions = train_only(df_train, model, tokenizer)
    if not args.test_only and not args.train_only:
        metrics, predictions = train_test(
            df_train, df_test, model, tokenizer, args.cross_validation)

    # dump metrics to dataframe
    df_metrics = pd.DataFrame(metrics)

    df_metrics = df_metrics.transpose()
    df_metrics = round(df_metrics, 2)
    df_metrics.to_csv(
        f"{args.output_name}/metrics_run_{args.output_identifier}.csv")
    # df_metrics.to_csv(f"{args.output_name}/metrics.csv")

    # save predictions
    dump(predictions,
         f"{args.output_name}/predictions_{args.output_identifier}.pkl")


if not args.test_only:
    # remove auto-saved runs directories
    rmtree(f"{args.output_name}/runs")
    # remove checkpoints
    checkpoints = glob.glob(f"{args.output_name}/checkpoint*")
    for entry in checkpoints:
        rmtree(entry)

    # dump args of training
    dump(args, f"{args.output_name}/arguments.pkl")
