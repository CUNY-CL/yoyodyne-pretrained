# Yoyodyne 🪀 pre-trained

This extends Yoyodyne to add support for using pre-trained transformer
encoder/decoder models from [Hugging Face](https://huggingface.co/).

It inherits most of the same features as Yoyodyne itself, except that the only
supported architecture consists of a pre-trained transformer encoder and a
pre-trained transformer decoder with a randomly-initialized cross-attention (à
la Rothe et al. 2020). Because these modules are pre-trained, there are no
architectural hyperparameters to set once one has determined which encoder and
decoder to "warm-start" from.

## TODO

* Datamodule
* Model
* CLI
* Sample configs
* Tests

## References

Rothe, S., Narayan, S., and Severyn, A. 2020. Leveraging pre-trained checkpoints
for sequence generation tasks. *Transactions of the Association for
Computational Linguistics* 8: 264-280.
