# Test TensorFlow nmt package

- You can check my codes and experiment setting here: https://github.com/lcy-seso/nmt/compare/format_and_add_precommit...lcy-seso:eval_default_device_placement

## Test Settings

- Test Eviornment:
  - TensorFlow r1.5 compiled by GCC 4.9, CUDA 8.0, cudnn 5.1, no NCCL support.
  - GTX Titan, 3 cards on one machine.

- Topology of the test model:
  - RNN encoder-decoder without attention.
    - source vocabulary size is: 7709
    - target vocabulary size is: 17191
    - embedding dimension: 512
    - 4 LSTM with 3 residual connections as encoder and 4 LSTM with 3 residual connections as decoder, their hidden dimensions are set to 512.
    - Use Adam as optimization algorithm.

## Some facts about the original implementations

Here I list some facts that potentially may affect the execution time performance.

### About I/O

TensorFlow Dataset API are used to read data from text files, which is claimed to be to be much more effiient than using `feeding_dict` and placeholder.

- It helps to build a pipeline to ensure that the GPU has never to wait for new data to come in. A lot of memory management is done for the user when using large file-based datasets.
- A background thread and an internal buffer are used to prefetch training samples from the input dataset ahead of the time they are requested.
- The default prefetch buffer is set to [1000 * batch_size](https://github.com/lcy-seso/nmt/blob/format_and_add_precommit/nmt/utils/iterator_utils.py#L95) to help overlap the input data preprocessing and model execution of a training step.
- The `output_buffer_size` canbe adjusted if data reading and computation cannot be overlapped.

### Bucket for reading training data

- TensorFlow requires sequences in one mini-batch have the same length even if dynamice rnn is used (sequences in different mini-batches can have different lengths but in one mini-batch, their lengths are required to be the same.) When reading training data, [buckets are used to reduce extra computations caused by padding](https://github.com/tensorflow/nmt/blob/master/nmt/utils/iterator_utils.py#L189).
- If `max source sequence length` is not set (by default, it is not set), the bucket width is set to 10 and `number_buckets` will be ignored.
- Training pairs with length `[0, bucket_width)` go to bucket 0, `length [bucket_width, 2 * bucket_width)` go to bucket 1, etc. How many buckets are formed will be determined dynamically according to the given trainning dataset.
- **Bucket can significantly help to speed up dynamic rnn training if sequences in on mini-batch have approximately the same length. But using bucket breaks the randomness which potentially harms the learning performance**.

### About the default model parallelism

The original open sourced implementation only implements the model parallelism. By default:

- If the source or target vocabulary size exceeds `50000`, the embedding will be [forced to be placed on `/cpu0`](https://github.com/tensorflow/nmt/blob/master/nmt/model_helper.py#L218).
- Round-robin placement of layers for multi-layer LSTM encoder/decoder is performed.
  - For example, if 3 GPU cards are avilable, a model with 4 LSTMs as encoder and 4 LSTMs as decoder is defined, then:
    - 4 LSTMs in encoder will be placed on: `/gpu0`, `/gpu1`, `/gpu2`, `/gpu0` seperately.
    - 4 LSTMs in decoder will be placed on: `/gpu0`, `/gpu1`, `/gpu2`, `/gpu0` seperately.
    - **It is better is the number of stacked RNN cell canbe evenly divided by the number of GPUs, otherwise, the workload will be unbalanced.**
- About the loss computation:
  - If only one GPU is avilable, the loss will be computed on CPU (including the pre-softmax projection and the softmax with cross entropy loss).
  - For multiple GPU training, because the multi-layer encoder uses a round-robin device placement method. For example, suppose 3 GPUs are aviliable and the last LSTM in encoder is placed on `/gpu1`, then the loss will be placed on `/gpu2` regardless how decoder is placed.

## Test the default model parallelism

- Disable data set shuffle so that run the experiment for multiple times will use the same data and the data are feeded to the network in the same order.
- Bucket = 5

### batch size = 128

|Number of GPU cards|Total batch size|Total time to run 50 mini-batch(s)|Samples per second|Speed-up ratio|
|--|--|--|--|--|
|1|128|23.22107|5.5122|
|2|128|32.17863|3.9778|0.72|
|3|128|25.51084|5.0175|0.9103|

> **The training speed becomes even slower by using multiple GPU cards**.

### batch size = 256

|Number of GPU cards|Total batch size|Total time to run 50 mini-batch(s)|Samples per second|Speed-up ratio|
|--|--|--|--|--|
|1|256|32.33249|6.1857|
|2|256|43.39386|4.6089|0.7451|
|3|256|32.28832|6.1942|0.9986|

## Problem for the default model parallel implementation

- **The workload for each GPU device tends to become unbalanced by using the default device placement method.**
- Therotically, placing multiple LSTMs on different devices has a certain degree parallism , but it seems that it still takes quite a lot of time for each operator waiting for their inputs are avilable.
