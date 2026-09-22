"""
Pooling sweep over data/prompt_paraphrases_100.csv: paraphrase-dataset version
of cos_measurement_pooling_privacy.py (same sampling, seeds, per-cell logic,
output columns and summary; see there for the metric definitions). Compares
mean / last-token / position-weighted pooling at one fixed hidden layer.

The input CSV has a fixed shape:

    id, original, paraphrase_synonyms, paraphrase_reordered, paraphrase_both

Outputs:
    <out_dir>/<save_suffix>.csv   one flat row per (id, rewrite_type, k, POOLING)
    stdout                        per-cell progress, then for every K a table
                                  with one line per pooling against the
                                  reference type (synonyms, as in the layer run).
"""
from cos_measurement_pooling_privacy import get_args, run
from cos_measurement_layers import REWRITE_COLS, TYPES, REFERENCE


if __name__ == "__main__":
    args = get_args(description="Pooling sweep of the cosine-similarity utility "
                                "over the paraphrase CSV",
                    csv_default="./data/prompt_paraphrases_100.csv",
                    suffix_default="paraphrases_cos_pooling")
    run(args, REWRITE_COLS, TYPES, REFERENCE)
