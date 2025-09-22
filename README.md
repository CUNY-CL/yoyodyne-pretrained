# Yoyodyne 🪀 Pretrained

[![PyPI
version](https://badge.fury.io/py/yoyodyne-pretrained.svg)](https://pypi.org/project/yoyodyne-pretrained)
[![Suported Python
versions](https://img.shields.io/pypi/pyversions/yoyodyne-pretrained.svg)](https://pypi.org/project/yoyodyne-pretrained)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CUNY-CL/yoyodyne-pretrained/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/CUNY-CL/yoyodyne-pretrained/tree/master)

Yoyodyne Pretrained provides sequence-to-sequence transduction with pretrained
transformer modules.

These models are implemented using [PyTorch](https://pytorch.org/),
[Lightning](https://lightning.ai/), and [Hugging Face
transformers](https://huggingface.co/docs/transformers/en/index).

## Philosophy

Yoyodyne Pretrained inherits many of the same features as Yoyodyne itself, but
limits itself to two types of pretrained transformers:

-   a pretrained transformer encoder and a pretrained transformer decoder with a
    randomly-initialized cross-attention (à la Rothe et al. 2020)
-   a T5 model

Because these modules are pretrained, there are few architectural
hyperparameters to set once one has determined which encoder and decoder to
warm-start from. To keep Yoyodyne as simple as possible, Yoyodyne Pretrained is
a separate library though it has many of the same features and interfaces.

## Installation

### Local installation

To install Yoyodyne Pretrained and its dependencies, run the following command:

    pip install .

### Google Colab

Yoyodyne Pretrained is also compatible with [Google
Colab](https://colab.research.google.com/) GPU runtimes.

1.  Click "Runtime" \> "Change Runtime Type".
2.  In the dialogue box, under the "Hardware accelerator" dropdown box, select
    "GPU", then click "Save".
3.  You may be prompted to delete the old runtime. Do so if you wish.
4.  Then install and run using the `!` as a prefix to shell commands.

## File formats

Other than YAML configuration files, Yoyodyne Pretrained operates on basic
tab-separated values (TSV) data files. The user can specify source, features,
and target columns. If a feature column is specified, it is concatenated (with a
separating space) to the source.

## Usage

The `yoyodyne_pretrained` command-line tool uses a subcommand interface with
four different modes. To see the full set of options available for each
subcommand, use the `--print_config` flag. For example:

    yoyodyne_pretrained fit --print_config

will show all configuration options (and their default values) for the `fit`
subcommand.

### Training (`fit`)

In `fit` mode, one trains a Yoyodyne Pretrained model from scratch. Naturally,
most configuration options need to be set at training time. E.g., it is not
possible to switch between different pretrained encoders after training a model.

This mode is invoked using the `fit` subcommand, like so:

    yoyodyne_pretrained fit --config path/to/config.yaml

#### Seeding

Setting the `seed_everything:` argument to some fixed value ensures a
reproducible experiment (modulo hardware non-determism).

#### Model architectures

##### Encoder-decoder models

In practice it is usually wise to tie the encoder and decoder parameters, as in
the following YAML snippet:

    ...
    model:
      class_path: yoyodyne_pretrained.models.EncoderDecoderModel
      init_args:
        model_name: google-bert/bert-base-multilingual-cased
        tie_encoder_decoder: true
        ...

##### T5 models

The following snippet shows a simple configuration T5 configuration using ByT5:

      class_path: yoyodyne_pretrained.models.T5Model
      init_args:
        model_name: google/byt5-base
        tie_encoder_decoder: true
        ...

#### Optimization

Yoyodyne Pretrained requires an optimizer and an learning rate scheduler. The
default optimizer is
[`torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html),
and the default scheduler is the inherited
[`yoyodyne.schedulers.Dummy`](https://github.com/CUNY-CL/yoyodyne/blob/d554eb891ac16a4cb741c507abc5e810321b7241/yoyodyne/schedulers.py#L9),
which keeps learning rate fixed at its initial value and takes no explicit
configuration arguments.

The following YAML snippet shows the use of the Adam optimizer with a
non-default initial learning rate and the inherited
[`yoyodyne.schedulers.WarmupInverseSquareRoot`](https://github.com/CUNY-CL/yoyodyne/blob/d554eb891ac16a4cb741c507abc5e810321b7241/yoyodyne/schedulers.py#L26)
LR scheduler:

    ...
    model:
        ...
        optimizer:
          class_path: torch.optim.Adam
          init_args:
            lr: 1.0e-5
        scheduler:
          class_path: yoyodyne.schedulers.WarmupInverseSquareRoot
          init_args:
            warmup_epochs: 10
        ...

#### Checkpointing

The
[`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
is used to control the generation of checkpoint files:

    ...
    checkpoint:
      filename: "model-{epoch:03d}-{val_accuracy:.4f}"
      mode: max
      monitor: val_accuracy
      verbose: true
      ...

Alternatively, one can specify a checkpointing that minimizes validation loss,
as follows:

    ...
    checkpoint:
      filename: "model-{epoch:03d}-{val_loss:.4f}"
      mode: min
      monitor: val_loss
      verbose: true
      ...

A checkpoint config must be specified or Yoyodyne Pretrained will not generate
any checkpoints.

#### Callbacks

The user will likely want to configure additional callbacks. Some useful
examples are given below.

The
[`LearningRateMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html)
callback records learning rates:

    ...
    trainer:
      callbacks:
      - class_path: lightning.pytorch.callbacks.LearningRateMonitor
        init_args:
          logging_interval: epoch
      ...

The
[`EarlyStopping`](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
callback enables early stopping based on a monitored quantity and a fixed
`patience`:

    ...
    trainer:
      callbacks:
      - class_path: lightning.pytorch.callbacks.EarlyStopping
        init_args:
          monitor: val_loss
          patience: 10
          verbose: true
      ...

#### Logging

By default, Yoyodyne Pretrained performs some minimal logging to standard error
and uses progress bars to keep track of progress during each epoch. However, one
can enable additional logging faculties during training, using a similar syntax
to the one we saw above for callbacks.

The
[`CSVLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html)
logs all monitored quantities to a CSV file. A sample configuration is given
below.

    ...
    trainer:
      logger:
        - class_path: lightning.pytorch.loggers.CSVLogger
          init_args:
            save_dir: /Users/Shinji/models
      ...

The
[`WandbLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html)
works similarly to the `CSVLogger`, but sends the data to the third-party
website [Weights & Biases](https://wandb.ai/site), where it can be used to
generate charts or share artifacts. A sample configuration is given below.

    ...
    trainer:
      logger:
      - class_path: lightning.pytorch.loggers.WandbLogger
        init_args:
          project: unit1
          save_dir: /Users/Shinji/models
      ...

Note that this functionality requires a working account with Weights & Biases.

#### Other options

Dropout probability and/or label smoothing are specified as arguments to the
`model`, as shown in the following YAML snippet.

    ...
    model:
      dropout: 0.5
      label_smoothing: 0.1
      ...

Decoding is performed with beam search if `model: num_beams: ...` is set to a
value greater than 1; the beam width ("number of beams") defaults to 5.

Batch size is specified using `data: batch_size: ...` and defaults to 32.

By default, training uses 32-bit precision. However, the `trainer.: precision:`
flag allows the user to perform training with half precision (`16`), or with
mixed-precision formats like `bf16-mixed` if supported by the accelerator. This
might reduce the size of the model and batches in memory, allowing one to use
larger batches, or it may simply provide small speed-ups.

There are a number of ways to specify how long a model should train for. For
example, the following YAML snippet specifies that training should run for 100
epochs or 6 wall-clock hours, whichever comes first.

    ...
    trainer:
      max_epochs: 100
      max_time: 00:06:00:00
      ...

### Validation (`validate`)

In `validation` mode, one runs the validation step over labeled validation data
(specified as `data: val: path/to/validation.tsv`) using a previously trained
checkpoint (`--ckpt_path path/to/checkpoint.ckpt` from the command line),
recording loss and other statistics for the validation set. In practice this is
mostly useful for debugging.

This mode is invoked using the `validate` subcommand, like so:

    yoyodyne_pretrained validate --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Evaluation (`test`)

In `test` mode, one computes accuracy over held-out test data (specified as
`data: test: path/to/test.tsv`) using a previously trained checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line); it differs from
validation mode in that it uses the `test` file rather than the `val` file.

This mode is invoked using the `test` subcommand, like so:

    yoyodyne_pretrained test --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Inference (`predict`)

In `predict` mode, a previously trained model checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line) is used to label
an input file. One must also specify the path where the predictions will be
written.

    ...
    predict:
      path: /Users/Shinji/predictions.conllu
    ...

This mode is invoked using the `predict` subcommand, like so:

    yoyodyne_pretrained predict --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

Many tokenizers, including the BERT tokenizer, are lossy in the sense that they
may introduce spaces not present in the input, particularly adjacent to
word-internal punctuation like dashes (e.g., *state-of-the-art*). Unfortunately,
there is little that can be done about this within this library, but it may be
possible to fix this as a post-processing step.

## Examples

See [`examples`](examples/README.md) for some worked examples including
hyperparameter sweeping with [Weights & Biases](https://wandb.ai/site).

## Testing

Given the size of the models, a basic integration test of Yoyodyne Pretrained
exceeds what is feasible without access to reasonably powerful GPU. Thus tests
have to be run locally rather than via cloud-based continuous integration
systems. The integration tests take roughly 30 minutes in total. To test the
system, run the following:

    pytest -vvv tests

### License

Yoyodyne Pretrained is distributed under an [Apache 2.0 license](LICENSE.txt).

## For developers

We welcome contributions using the fork-and-pull model.

### Releasing

1.  Create a new branch. E.g., if you want to call this branch "release":
    `git checkout -b release`
2.  Sync your fork's branch to the upstream master branch. E.g., if the upstream
    remote is called "upstream": `git pull upstream master`
3.  Increment the version field in [`pyproject.toml`](pyproject.toml).
4.  Stage your changes: `git add pyproject.toml`.
5.  Commit your changes: `git commit -m "your commit message here"`
6.  Push your changes. E.g., if your branch is called "release":
    `git push origin release`
7.  Submit a PR for your release and wait for it to be merged into `master`.
8.  Tag the `master` branch's last commit. The tag should begin with `v`; e.g.,
    if the new version is 3.1.4, the tag should be `v3.1.4`. This can be done:
    -   on GitHub itself: click the "Releases" or "Create a new release" link on
        the right-hand side of the Yoyodyne GitHub page) and follow the
        dialogues.
    -   from the command-line using `git tag`.
9.  Build the new release: `python -m build`
10. Upload the result to PyPI: `twine upload dist/*`

## References

Rothe, S., Narayan, S., and Severyn, A. 2020. Leveraging pre-trained checkpoints
for sequence generation tasks. *Transactions of the Association for
Computational Linguistics* 8: 264-280.

(See also [`yoyodyne-pretrained.bib`](yoyodyne-pretrained.bib) for more work
used during the development of this library.)
