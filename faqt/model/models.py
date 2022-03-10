import numpy as np
from .embeddings import model_search_word, model_search, cs_nearest_k_percent_average


class FAQScorer:
    def __init__(
        self,
        w2v_model,
        glossary=None,
        hunspell=None,
        tags_guiding_typos_wv=None,
    ):
        self.w2v_model = w2v_model
        self.glossary = glossary
        self.hunspell = hunspell
        self.tags_guiding_typos_wv = tags_guiding_typos_wv

    def fit(self, faqs):
        for faq in faqs:
            faq.faq_tags_wvs = {
                tag: model_search_word(tag, self.model, self.glossary)
                for tag in faq.faq_tags
            }

        self.faqs = faqs
        return self

    def score(
        self,
        message,
        n_top_matches,
        scoring_function=cs_nearest_k_percent_average,
        **scoring_func_kwargs
    ):

        top_matches_list = []
        scoring = {}
        inbound_vectors, inbound_spellcorrected = model_search(
            message,
            model=self.w2v_model,
            glossary=self.glossary,
            hunspell=self.hunspell,
            tags_guiding_typos_wv=self.tags_guiding_typos_wv,
            return_spellcorrected_text=True,
        )

        if len(inbound_vectors) == 0:
            return top_matches_list, scoring, ""

        scoring = get_faq_scores_for_message(inbound_vectors, self.faqs)
        top_matches_list = get_top_n_matches(scoring, n_top_matches)

        return top_matches_list, scoring, inbound_spellcorrected


def get_faq_scores_for_message(
    inbound_vectors, faqs, scoring_function, **scoring_func_kwargs
):

    scoring = {}
    for faq in faqs:
        scoring[faq.faq_id] = {}
        scoring[faq.faq_id]["faq_title"] = faq.faq_title
        scoring[faq.faq_id]["faq_content_to_send"] = faq.faq_content_to_send
        scoring[faq.faq_id]["tag_cs"] = {}

        for tag, tag_wv in faq.faq_tags_wvs.items():
            if tag_wv is not None:
                scoring[faq.faq_id]["tag_cs"][tag] = scoring_function(
                    inbound_vectors, tag_wv, **scoring_func_kwargs
                )
            else:
                scoring[faq.faq_id]["tag_cs"][tag] = 0

        cs_values = list(scoring[faq.faq_id]["tag_cs"].values())
        scoring[faq.faq_id]["overall_score"] = (min(cs_values) + np.mean(cs_values)) / 2

    return scoring


def get_top_n_matches(scoring, n_top_matches):
    matched_faq_titles = set()

    # Sort and copy over top matches
    top_matches_list = []
    for id in sorted(scoring, key=lambda x: scoring[x]["overall_score"], reverse=True):
        if scoring[id]["faq_title"] not in matched_faq_titles:
            top_matches_list.append(
                (scoring[id]["faq_title"], scoring[id]["faq_content_to_send"])
            )
            matched_faq_titles.add(scoring[id]["faq_title"])

        if len(matched_faq_titles) == n_top_matches:
            break

    return (top_matches_list,)
