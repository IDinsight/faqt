# Running Tests

## Setup

In order to run tests locally, you need the pretrained w2v model binary. This should
be placed in `pretrained_wv_models`. The w2v binary is loaded in
`conftest.py` once per test session.

At present, it loads the Google New Model and looks for a file called
`GoogleNews-vectors-negative300-prenorm.bin` which is the only w2v model that has
been tested.

## Run tests

Run `pytest` in the `tests` directory or it's parent.
