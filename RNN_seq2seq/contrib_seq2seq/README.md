# API guide for [tf.contrib.seq2seq](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/seq2seq/)

- As of r1.1
- tf.contrib.seq2seq has similar API with [tf-seq2seq](https://google.github.io/seq2seq/)
- All codes in readme are Pythonic pseudocodes

## Classes

### Helper
- Used as an attribute of [Decoder](#decoder) class
- Replaces wrapper functions (ex. `EmbeddingWrapper`, `OutputProjectionWrapper`, `AttentionWrapper`) used in [legacy_seq2seq](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py) functions
- Defined in [helper.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/seq2seq/python/ops/helper.py)

#### `__init__(inputs, sequence_length, time_major=False)`
```
input_tas = TensorArray.unstack(inputs)
```

#### `initialize()`
```
finished = [False, False, ...] # [batch_size, ]
if all(finished):
  next_inputs = zero_inputs # zero-tensor with same shape
else:
  next_inputs = input_tas[0]
return finished, next_inputs
```

#### `sample(time, outputs)`
```
sample_ids = tf.argmax(outputs, axis=-1, dtype=tf.int32)
```

#### `next_inputs(time, outputs, state)`
```
time += 1
finished = (time > sequence_length) # check if each batch is completed
if all(finished):
  next_inputs = zero-tensor with same shape
else:
  next_inputs = input_tas[time]
return finished, next_inputs, state
```

### DecoderOutput
- Wraps [Decoder](#decoder)'s attributes such as `output_size`, `output_dtype`
- [tf.contrib.seq2seq](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py#L40) ver. (`BasicDecoderOutput` in [basic_decoder.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py))
```
class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
```
- [tf-seq2seq](https://github.com/google/seq2seq/blob/master/seq2seq/) ver. (`DecoderOutput` in [rnn_decoder.py](https://github.com/google/seq2seq/blob/master/seq2seq/decoders/rnn_decoder.py))
```
class DecoderOutput(
    namedtuple("DecoderOutput", ["logits", "predicted_ids", "cell_output"]))
```

### Decoder
- Main decoder class
- Used as an attribute of [dynamic_decode](#dynamic_decode)
- `basic_decoder` is defined in [basic_decoder.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py)

#### `__init__(cell, helper, initial_state, output_layer=None)`
- output_layer: an instance of [`tf.layers.Layer`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/core.py)
- ex) `output_layer = tf.layers.core.Dense(output_layer_depth, use_bias=False)`

#### `_rnn_output_size()`
```
if output_layer is None:
  return size
else:
  return output_layer._compute_output_shape[1:]
```

#### `output_size`
```
return DecoderOutput(_rnn_output_size(), TensorShape([]))
```

#### `initialize()`
```
return decoder._helper.initialize() + (decoder._initial_state, )
```

#### `step(time, inputs, state)`
```
cell_outputs, cell_state = cell(inputs, state)
if output_layer is not None:
  cell_outputs = output_layer(cell_outputs)
sample_ids = helper.sample(time, cell_outputs) # sometimes cell_state is needed
finished, next_inputs, next_state = helper.next_inputs(time, cell_outputs, cell_state, sample_ids)
outputs = DecoderOutput(cell_outputs, sample_ids)
return outputs, next_state, next_inputs, finished
```

## Functions

### dynamic_decode
- Perform dynamic decoding with [Decoder](#decoder)
- Defined in [decoder.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/seq2seq/python/ops/decoder.py)

#### Args:
- decoder: a [Decoder](#decoder) instance
- output_time_major=False
- impute_finished=False
- maximum_iterations=32
- swap_memory=False

#### Pseudocode
```
finished, inputs, state = decoder.initialize()

time = 0

outputs_ta = TensorArray(size=decoder.output_size)

while not all(finished):
  outputs, state, inputs, finished = decoder.step(time, inputs, state)

  if maximum_iterations is not None and time + 1 >= maximum_iterations:
      finished = True

  # if finished!
  # => zero out all remaining outputs
  if impute_finished:
    outputs = zero_outputs # zero-tensor with same shape

  outputs_ta[time] = outputs

  time += 1

final_outputs = outputs_ta.stack()
final_state = state

# time_major => batch_major
# [max_dec_len x batch_size x hidden_size] => [batch_size x max_dec_len x hidden_size]
if not output_time_major:
  final_outputs = tf.transpose(final_outputs, [1,0,2])

return final_outputs, final_state

```
