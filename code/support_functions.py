import unicodedata
from nltk import ngrams
import numpy as np
import pandas as pd
from tqdm import tqdm
import regex
import logging
import spacy
from typing import List, Optional, Tuple, Dict

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

logging = logging.getLogger(__name__)

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "uk": "en_core_web_sm",
    "wales": "en_core_web_sm",
    "scotland": "en_core_web_sm",
    "nireland": "en_core_web_sm",
    "es": "es_core_news_sm",
    "catalan": "es_core_news_sm",
    "basque": "es_core_news_sm",
    "gr": "el_core_news_sm",
}



def get_head_tail(df, mode='retweet_count', head_perc=0.95, tail_perc=0.5):
    """
    Given a dataframe `df` return the head and tail of the dataframe according to mode provided.

    Args:
        df (pd.DataFrame): Original dataframe
        mode (Optional[str], optional): Metric to be used to get head/tail. Possible values: `retweet_follower_ratio,
                                        retweets_in_relation_to_average, retweet_count`. Defaults to 'retweet_count'.
        head_perc (Optional[float], optional): Percentile threshold to consider entries in head. Defaults to `0.95`.
        tail_perc (Optional[float], optional): Percentile threshold to consider entries in tail. Defaults to `0.35`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: head, tail of the dataframe
    """
    top_threshold_retweets = df[mode].quantile(head_perc)
    bot_threshold_retweets = df[mode].quantile(tail_perc)
    idx_top_retweets = df[df[mode] >= top_threshold_retweets].index
    idx_bot_retweets = df[df[mode] <= bot_threshold_retweets].index

    head = df.loc[idx_top_retweets]
    tail = df.loc[idx_bot_retweets]

    return head, tail


def calc_in_relation_to_average(x) -> float:
    """
    Support function to calculate the "in_relation_to_average" statistic. i.e How much a tweet
    has been retweeted given the avg of the retweets of the user. E.g user's X retweet_count
    average is 100. One of his tweets is retweeted 80 times then this tweets
    "in_relation_to_average" score would be 80.

    Args:
        x (pd.Dataframe row): Row of a dataframe. Must contain "retweet_count", "avg_retweets"
                              columns.

    Returns:
        (float): The "in_relation_to_average" statistic.
    """
    if x['avg_retweets'] > 0:
        return (100 * x['retweet_count']) / x['avg_retweets']
    else:
        return 0


def strip_welsh_stopwords(s: List[str]) -> List[str]:
    """
    Remove welsh stopwords


    Args:
        s (List[str]): list of tokens

    Returns:
        List[str]: list of tokens without stopwords
    """

    welsh_stopwords = {'mae', 'wedi', 'yr', 'n', 'ac', 'ei', 'ein', 'ni', 'gan',
                       'dros', 'fy', 'gyda', 'mewn', 'chi', 'â', 'sydd', 'yma', 'bydd',
                       'ond', 'bod', 'yn', 'ar', 'r'}

    return [token for token in s if token not in welsh_stopwords]


def remove_handles(text: str) -> str:
    """
    Remove Twitter username handles from text.

    Args:
        text (str): Input text.

    Returns:
        str: Output text without handles
    """
    pattern = regex.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)"
    )
    # Substitute handles with ' @username ' to ensure that text on either side of removed handles are
    # tokenized correctly
    return pattern.sub(" @username ", text)


def strip_accents_and_lowercase(s: str) -> str:
    """
    Strip accents for greek.

    Args:
        s (str): Input text.

    Returns:
        str: Output text without accents.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()


def tokenize_text(documents: List[str], nlp_model: str, strip_handles=True,
                  ngram: Optional[int] = None) -> List[List[str]]:
    """
    Tokenize documents using a spacy model.
    We do not consider: stopwords, punctuation, urls, numbers, 'amp' charachter, tokens with 1 character. 

    Args:
        documents (List[str]): A list of documents to be tokenized
        nlp_model (str): A str code to select a spacy model (see SPACY_MODELS)
        strip_handles (bool, optional): Whether to remove handles from docs. Defaults to True.
        ngram (Optional[int], optional): Ngrams range to generate. Defaults to None.

    Returns:
        List[List[str]]: Tokenized documents. A list of lists of tokens.
    """
    results = []
    nlp = spacy.load(SPACY_MODELS[nlp_model])
    nlp.add_pipe("emoji", first=True)
    logging.info(f"Using {SPACY_MODELS[nlp_model]}")

    if strip_handles:
        documents = [remove_handles(x)
                     for x in tqdm(documents, desc="Removing handles")]

    # strip greek accents
    if nlp_model == "gr":
        documents = [strip_accents_and_lowercase(x) for x in tqdm(
            documents, desc="Stripping accents.")]

    # tokenizer pipeline
    for doc in tqdm(nlp.pipe(documents, batch_size=2000, n_process=4), total=len(documents), desc='Tokenizing text'):
        tokens = []
        for token in doc:
            if ((token.is_stop == False) and (token.is_punct == False) and (token.like_url == False)
                    and (token.is_space == False) and (token.like_num == False) and (token.lemma_ != "amp")
                    and (len(token) > 2)) or (token._.is_emoji):
                tokens.append(token.text.lower())

        results.append(tokens)

    # remove welsh punctuation
    if nlp_model == "wales":
        results = [strip_welsh_stopwords(x) for x in tqdm(
            results, desc="Removing welsh stopwords")]
        pass

    if ngram is not None:
        results = [y + [" ".join(x) for x in list(ngrams(y, ngram))]
                   for y in results]
    return results
