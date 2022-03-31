Usage
=====

Installation
------------


Eventually `FAQT` will be open-source an available from pypi. In the meantime, you can use `pip` to install it from github:

.. code-block:: console

    $ pip install  git+https://@github.com/IDinsight/faqt.git@main

It will prompt you for your username and password since the repository is private.

Development
~~~~~~~~~~~

If you will be developing on :code:`FAQT`, you should clone the repo and run :code:`pip install -e .` within the repo.


Basic Usage
-----------
:emphasis:`FAQT` has three components:

1. :code:`faqt.preprocessing` - contains functions to preprocess free text and convert them to tokens before sending it to the model
2. :code:`faqt.model` - contains classes with functions to load or update sets of tags, and score messages against the sets of tags.
3. :code:`faqt.scoring_functions` - contains functions that take a set of tokens from a message and a set of tags and return a similarity score.

Here's a basic example that uses all three

.. code-block:: python
   :dedent: 

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





