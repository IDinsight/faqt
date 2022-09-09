# Running Tests

## Setup

In order to run tests locally, you need 
- pretrained w2v model binary in `data/pretrained_wv_models`
  - The w2v model is loaded in `conftest.py` once per test session.
  - At present, it loads the Google New Model and looks for a file called
`GoogleNews-vectors-negative300-prenorm.bin` which is the only embeddings model that has
been tested.
- HuggingFace `transformers` sequence classification model (for question-answer pair scoring), in `data/sequence_classification`
  - At present, the tests looks for a directory called `huggingface_model`.


## Run tests

Run `pytest` in the `tests` directory or it's parent.
