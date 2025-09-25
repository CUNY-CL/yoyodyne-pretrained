SHELL := /bin/bash

all: train.tsv \
	dev.tsv \
	test.tsv \
	byt5_expected.test \
	mbert_tied_expected.test

# Data from the CoNLL-SIGMORPHON 2017 shared task on morphological reinflection.

train.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/english-train-medium > "$@"

dev.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/english-dev > "$@"

test.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/answers/task1/english-uncovered-test > "$@"

%_expected.test: train.tsv dev.tsv test.tsv ../configs/%.yaml
	$(eval TMPDIR := $(shell mktemp -d))	
	yoyodyne_pretrained \
		fit \
		--checkpoint ../configs/checkpoint.yaml \
		--data ../configs/data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../configs/$*.yaml" \
		--seed_everything 49 \
		--trainer ../configs/trainer.yaml
	yoyodyne_pretrained \
		test \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../configs/data.yaml \
		--data.test test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../configs/$*.yaml" \
		--trainer.enable_progress_bar false \
		> "$@"
	rm -rf "$(TMPDIR)"
