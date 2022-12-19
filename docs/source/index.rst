.. FAQT documentation master file, created by
   sphinx-quickstart on Mon Mar 28 11:37:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. module:: faqt
Welcome to FAQT's documentation!
================================
**FAQT** is a Python library to match text to a set of keywords.

How it works
------------
Say you have 3 sets of keyword or tags, each representing some predefined content or a topic:

* 1: {"river", "lake", "mountains"}
* 2: {"movie", "dinner", "walk"}
* 3: {"aircraft", "terminal", "boarding"}

And you want to see how well the messages "i went camping in the woods last weekend" matches the tags.

FAQT will return similarity scores for the message against each set of tags:
:code:`{1: 0.7, 2: 0.3, 3: 0.4}`.

Check out the :doc:`usage` section for further information.

.. note::
   This project is under active development and API may change substantially.


Usage
--------
.. toctree::
   :maxdepth: 4

   usage

Contents
--------
.. toctree::
   :maxdepth: 4

   faqt.preprocessing 
   faqt.model.faq_matching
   faqt.scoring
   faqt.model.urgency_detection

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`





