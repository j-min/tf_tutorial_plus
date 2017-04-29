# Architecture Guidelines

##### Separating your modules in right way can improve 'reproducibility' and 'readability' of your codes significantly.
#### After you walkthrough my tutorials, I would suggest implementing your own models in separate files.

##### For example,

```
project_dir/
  -- data_dir/
      -- bunch of data files..
  -- utils.py
  -- models.py
  -- train.py
  -- evaluate.py
  -- config.py
  -- hpsearch.py
```

### `utils.py`

```
import os
import re
import Counter

tokenizer = ...
```

`models.py` (or `base_model.py` + `layers.py` + `models.py`)

```
import tensorflow as tf
from utils import tokenizer, ..

class Seq2SeqModel(..)
...
```

### `train.py`

```
from model import Seq2SeqModel

model = Seq2SeqMoel(..)
model.build(..)
model.train(..)
```

### `evaluate.py`

```
from model import Seq2SeqModel

model = Seq2SeqMoel(..)
model.build(..)
model.evaluate(..)
```

### (optional 1) `config.py`
- path/hyperparameter configurations
- For example,
	- Non-Python extensions (`ConfigParser` + `.ini / .yaml`...)
	- Python class / dictionary
	- Argument parsers (`argparse`, `tf.app.flags`)
	- Combined parsers (`configargparse`)

### (optional 2) `hpsearch.py`
- meta-module for searching the best hyperparameter combination of your model.
- For example,
	- `sklearn.model_selection`
