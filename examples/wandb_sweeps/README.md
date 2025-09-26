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
[`configs/byt5_grid.yaml`](configs/byt5_grid.yaml), specifies the hyperparameter
grid (it may also contain constant values, if desired). The second file,
[`configs/tune.yaml`](configs/tune.yaml), specifies any constants needed during
the sweep, such as trainer arguments or data paths.

    # Creates a sweep; save the sweep ID as ${SWEEP_ID} for later.
    wandb sweep \
        --entity "${ENTITY}" \
        --project "${PROJECT}" \
        configs/byt5_grid.yaml
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

- See the [Yoyodyne sweeps documentation
  here](https://github.com/CUNY-CL/yoyodyne/blob/master/examples/wandb_sweeps/README.md).
