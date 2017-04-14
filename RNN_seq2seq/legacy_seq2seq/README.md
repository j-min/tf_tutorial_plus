# API guide for [tf.contrib.legacy_seq2seq](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py)

### I highly recommend skimming the original codes above before you start reading my codes.

### All codes have same goal (Basic Chatbot with seq2seq) with different APIs

### Example
- 'Hi What is your name?'
=> 'Hi this is Jaemin.'
- 'Nice to meet you!
=> 'Nice to meet you too!'

### The attention mechanism used here is of [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449).

### If you are already familiar with seq2seq codes, try replacing codes here with high-level NLP packages such as [nltk](http://www.nltk.org/index.html), [spacy](https://spacy.io/) or [konlpy](http://konlpy.org/en/v0.4.4/)!
- You can start with [nltk.FreqDist](http://www.nltk.org/book/ch02.html)

### Contents

- Preprocess: Vocabulary / tokenizer
- Word Embeddings
- Basic seq2seq model `BasicRNNCell` with `static_rnn`
- Training / Decode / Plot
- `XXX_Wrappers`
- `rnn_decoder`
- `embedding_rnn_decoder`
- `embedding_rnn_seq2seq`
- `attention_decoder`
- `embedding_attention_decoder`
- `embedding_attention_seq2seq`
