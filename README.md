# Negativity spreads faster: A large-scale multilingual twitter analysis on the role of sentiment in political communication
Data and code accompanying the paper [Negativity spreads faster: A large-scale multilingual twitter analysis on the role of sentiment in political communication](https://www.sciencedirect.com/science/article/pii/S2468696423000010).

In specific:
- the code used for the training of our models (./code/finetune_models.py and ./code/finetune_multi_cv.py)
- a Jupyter Notebook containing the major parts of our analysis (./code/analysis.ipynb)
- the model that was selected and used for the sentiment analysis.
- the manually annotated data used for training are shared (./data/annotation/).
- the ids of tweets that were used in our analyis and control experiments (./data/main/ & ./data/control)
- names, parties and handles of the MPs that were tracked (./data/mps_list.csv).


## Annotated Data (./data/annotation/)
- One folder for each language (English, Spanish, Greek).
- In each directory there are three files:
    1. *_900.csv  contains the 900 tweets that annotators labelled individually (300 tweets each annotator).
    2. *_tiebreak_100.csv contains the initial 100 tweets all annotators labelled. 'annotator_3' indicates the annotator that was used as a tiebreaker.
    3. *_combined.csv contains all tweets labelled for the language.


## Model
Our model, xlm-twitter-politics-sentiment, along with a small tutorial on how to use it can be found in [huggingface.co](https://huggingface.co/cardiffnlp/xlm-twitter-politics-sentiment).

The model is based on the implementation of 'cardiffnlp/twitter-xlm-roberta-base-sentiment' while being further finetuned on the annotated dataset.

### Example usage
```
from transformers import AutoModelForSequenceClassification, pipeline

model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/xlm-twitter-politics-sentiment')
sentiment_analysis_task = pipeline("sentiment-analysis", model=model, tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

sentiment_analysis_task('Today is a good day')
Out: [{'label': 'Positive', 'score': 0.978614866733551}]
```

## Reference paper

For more details, please check the [reference paper](https://www.sciencedirect.com/science/article/pii/S2468696423000010). If you use the data contained in this repository for your research, please cite the paper using the following `bib` entry:

```
@article{antypas2023negativity,
  title={Negativity spreads faster: A large-scale multilingual twitter analysis on the role of sentiment in political communication},
  author={Antypas, Dimosthenis and Preece, Alun and Camacho-Collados, Jose},
  journal={Online Social Networks and Media},
  volume={33},
  pages={100242},
  year={2023},
  publisher={Elsevier}
}
```
