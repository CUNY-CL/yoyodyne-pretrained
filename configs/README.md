The files in this directory are example YAML configuration files showing
Yoyodyne's use with various pre-trained encoders.

Since the model builds its own cross-attention, it is possible to mix and match
encoders and decoders. We are currently recommending the use of multilingual BERT, with or without parameter-tying.

Pre-trained encoders:

-   [mBERT
    (`google-bert/bert-base-multilingual-cased`)](https://huggingface.co/google-bert/bert-base-multilingual-cased)
