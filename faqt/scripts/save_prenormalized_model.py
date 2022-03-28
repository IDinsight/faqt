import os
import sys

from src.model.embeddings import load_wv_pretrained_bin

if __name__ == "__main__":
    os.chdir(sys.path[0])

    model = load_wv_pretrained_bin(
        "pretrained_wv_models", "GoogleNews-vectors-negative300.bin"
    )
    model.init_sims(replace=True)
    model.save_word2vec_format(
        "../data/pretrained_wv_models/GoogleNews-vectors-negative300-prenorm.bin",
        binary=True,
    )
