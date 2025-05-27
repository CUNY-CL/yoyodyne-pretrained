This directory contains small "toy" samples drawn from CoNLL-SIGMORPHON 2017
shared task on inflection for English and Russian for
`yoyodyne_pretrained_test.py`. The `_train.tsv` files are used to train and
validate the model. The `_expected.tsv` files are the result of applying the
model to the training data, and the `_expected.test` files give accuracy
results. Each file contains 1,000 sentences.

The following commands, run in the root directory, were used to generate the
expected data files:

    for LANGCODE in en ru; do
        yoyodyne_pretrained fit \
            --config=tests/testdata/mbert_config.yaml \
            --data.train="tests/testdata/${LANGCODE}_train.tsv" \
            --data.val="tests/testdata/${LANGCODE}_train.tsv"
        yoyodyne_pretrained predict \
            --ckpt_path=lightning_logs/version_0/checkpoints/last.ckpt \
            --config=tests/testdata/mbert_config.yaml \
            --data.predict="tests/testdata/${LANGCODE}_train.tsv" \
            --prediction.path="tests/testdata/${LANGCODE}_expected.txt"
        yoyodyne_pretrained test \
            --ckpt_path=lightning_logs/version_0/checkpoints/last.ckpt \
            --config=tests/testdata/mbert_config.yaml \
            --data.test="tests/testdata/${LANGCODE}_train.tsv" \
            > "tests/testdata/${LANGCODE}_expected.test"
        rm -rf lightning_logs
    done
