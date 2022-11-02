# FAQT

`FAQT` matches messages to sets of tags and return similarity scores.

[![Coverage Status](https://coveralls.io/repos/github/IDinsight/faqt/badge.svg?t=OJGPBt)](https://coveralls.io/github/IDinsight/faqt?branch=main)
![Unit Tests](https://github.com/IDinsight/faqt/actions/workflows/tests.yml/badge.svg)

## How it works

Say you have 3 sets of keyword or tags, each representing some predefined content or a topic:

-   1: {"river", "lake", "mountains"}
-   2: {"movie", "dinner", "walk"}
-   3: {"aircraft", "terminal", "boarding"}

And you want to see how well the messages "i went camping in the woods last weekend" matches the tags.

FAQT will return similarity scores for the message against each set of tags:
`{1: 0.7, 2: 0.3, 3: 0.4}`.

## Installation

Install Cython. Some of FAQT's dependencies require Cython to build.
```bash
pip install Cython>=0.29.32
```
Install FAQT. Eventually `FAQT` will be open-source an available from pypi. In the meantime, you can use `pip` to install it from github:
```bash
pip install git+https://@github.com/IDinsight/faqt.git@main
```
It will prompt you for your username and password since the repository is private.

### Extended installation
If you want to use a BERT-based model such as `faqt.model.faq_matching.bert.QuestionAnswerBERTScorer`, or use the `faqt.preprocessing.tokens.CustomHunspell` class, install faqt with its `extended` optional dependencies:
```bash
pip install git+https://@github.com/IDinsight/faqt.git@main[extended]
```

### Development

If you will be developing on `FAQT`, you should clone the repo and run `pip install -e '.[dev]'` within the repo.

## Basic Usage

`FAQT` has three components:

1. `faqt.preprocessing` - contains functions to preprocess free text and convert them to tokens before sending it to the model
2. `faqt.model` - contains classes with functions to load or update sets of tags, and score messages against the sets of tags.
3. `faqt.scoring_functions` - contains functions that take a set of tokens from a message and a set of tags and return a similarity score.

Here's a basic example that uses all three

```
from faqt.preprocessing import preprocess_text
from faqt.scoring_functions import cs_nearest_k_percent_average
from faqt.model import FAQScorer

## Must load and send a w2v binary using gensim
w2v_model = load_wv_model()
tags = {
            1: {"river", "lake", "mountains"},
            2: {"movie", "dinner", "walk"}
            3: {"aircraft", "terminal", "boarding"}
       }
message = "i went camping in the woods last weekend"

## Create model instance
model = FAQScorer(
            w2v_model,
            scoring_function= cs_nearest_k_percent_average,
            k=10,
            floor=1
        )

## Set the FAQs (sets of tags) for this model
model.set_tags(tags)

## Preprocess message
message_tokens = preprocess_text(message, {}, 10)

## Score message
scores = model.score(message_tokens)
```
