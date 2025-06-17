This directory contains small "toy" samples drawn from CoNLL-SIGMORPHON 2017
shared task on inflection. Currently only English is supported though the system
can easily be generalized. The `_train.tsv` files are used to train and validate
the model. The `_expected.tsv` files are the result of applying the model to the
training data, and the `_expected.test` files give accuracy results. Each file
contains 1,000 sentences.

The following commands, run in the root directory, were used to generate the
expected data files:

    yoyodyne_pretrained fit \
        --config=tests/testdata/mbert.yaml \
        --data.train=tests/testdata/en_train.tsv \
        --data.val=tests/testdata/en_train.tsv
    yoyodyne_pretrained predict \
        --ckpt_path=lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/mbert.yaml \
        --data.predict=tests/testdata/en_train.tsv \
        --prediction.path=tests/testdata/en_expected.txt
    yoyodyne_pretrained test \
        --ckpt_path=lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/mbert.yaml \
        --data.test=tests/testdata/en_train.tsv \
        > tests/testdata/en_expected.test
    rm -rf lightning_logs
