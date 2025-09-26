# W&B Sweeps

This directory contains documentation on running a hyperparameter sweep with
[Weights & Biases](https://wandb.ai/site) and software from
[Yoyodyne](https://github.com/CUNY-CL/yoyodyne).

## Usage

Execute the following to create and run the sweep; here `${ENTITY}` and
`${PROJECT}` are assumed to be pre-specified environmental variables.

In the following example, targeting the English data for the CoNLL-SIGMORPHON
2017 shared task on morphological generation, we have two separate YAML
configuration files prepared. The first file,
[`configs/mbert_grid.yaml`](configs/mbert_grid.yaml), specifies the
hyperparameter grid (it may also contain constant values, if desired). A similar
file ([`configs/byt5_grid.yaml`](configs/byt5_grid.yaml)) is available for ByT5.
The second file, [`configs/tune.yaml`](configs/tune.yaml), specifies any
constants needed during the sweep, such as trainer arguments or data paths.

    # Creates a sweep; save the sweep ID as ${SWEEP_ID} for later.
    wandb sweep \
        --entity "${ENTITY}" \
        --project "${PROJECT}" \
        configs/mbert_grid.yaml
    # Runs the sweep itself using hyperparameters from the the sweep and
    # additional fixed parameters from a Yoyodyne Pretrained config file.
    # The --command argument tells the sweep to use Yoyodyne Pretrained rather
    # than vanilla Yoyodyne.
    yoyodyne_sweep \
        --command "yoyodyne_pretrained" \
        --entity "${ENTITY}" \
        --project "${PROJECT}" \
        --sweep_id "${SWEEP_ID}" \
        --count "${COUNT}" \
        --config configs/tune.yaml

Then, one can retrieve the results as follows:

1.  Visit the following URL:
    `https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}`

2.  Switch to "table view" by either clicking on the spreadsheet icon in the top
    left or typing Ctrl+J.

3.  Click on the downward arrow link, select "CSV Export", then click "Save as
    CSV".

Or, to get the hyperparameters for a particular run, copy the "Run path" from
the run's "Overview" on W&B, and then run:

    yoyodyne_hyperparameters "${RUN_PATH}"

## Additional tips

-   Consider also running Bayesian search (`method: bayes`) instead of random
    search; see
    [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#configuration-keys)
    for more information.
-   When running a sweep you must provide the `--entity`, `--project` and
    `--sweep_id`; arguments not being tuned should be passed using the
    `--config` file but note that hyperparameters set by the sweep will override
    those specified in the `--config`.
-   By default search runs indefinitely, i.e., until the sweep command
    terminates. To specify a fixed number of samples, provide the `--count`
    argument.
-   For more information about W&B sweeps, [read
    here](https://docs.wandb.ai/guides/sweeps).
