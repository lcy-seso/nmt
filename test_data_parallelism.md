# Test data parallelism for NMT model

## Codes

- Use this codes: https://github.com/lcy-seso/nmt/compare/format_and_add_precommit...lcy-seso:add_data_parallelism
  - commit : https://github.com/lcy-seso/nmt/commit/2d73467680c9944b0f90b6df52efb580476c81ef

> The current data parallelism implementaion can still be further optimized.

### Notes about current implementation

1. The entire model are run GPU.
1. Each GPU card has a entire model replica.
1. Input mini-batch is evenly divided into multiple GPU cards by using [tf.split](https://www.tensorflow.org/api_docs/python/tf/split). So please garantee the batch size (in count of sequnce pair, not words) can be evenly divided by the number of GPU cards.
1. Forward and backward computation for one operator are forced to be placed on the same device.
1. `/gpu:0` is used as the main card to merge the gradients computed also by other GPU cards.
1. Parameter updates are only performed on `/gpu:0`.
1. `/gpu:0` use `gather` and `broadcast` to collect gradients and send out the updated parameters.

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

- Disable dataset shuffle, so that running the experiment for multiple times to run multiple batches will use the same training data and the data are feeded to the network in the same order.
- Run 50 mini-batches, and calculate the time that how many samples processed every second.

## Test results

1. bucket number = 1

    |Number of GPU cards|Total batch size|Total time to run 50 mini-batch(s)|Samples per second|Speed-up ratio|
    |--|--|--|--|--|
    |1|100|37.10221|2.6953|
    |2|100 * 2|49.95347|4.0037|1.48|
    |3|100 * 3|57.5135|5.2162|1.9353|

    |Number of GPU cards|Total batch size|Total time to run 50 mini-batch(s)|Samples per second|Speed-up ratio|
    |--|--|--|--|--|
    |1|200|57.15430|3.4993|
    |2|200 * 2|75.13103|5.3240|1.5218|
    |3|200 * 3|82.41376|7.2803|2.0805|

1. bucket number = 5

    |Number of GPU cards|Total batch size|Total time to run 50 mini-batch(s)|Samples per second|Speed-up ratio|
    |--|--|--|--|--|
    |1|100|22.09906|4.5251|
    |2|100 * 2|30.46061|6.5659|1.4510|
    |3|100 * 3|35.71756|8.3992|1.8561|

    |Number of GPU cards|Total batch size|Total time to run 50 mini-batch(s)|Samples per second|Speed-up ratio|
    |--|--|--|--|--|
    |1|200|33.35367|5.9963|
    |2|200 * 2|45.31347|8.8274|1.4721|
    |3|200 * 3|49.95583|12.0101|2.003|

## Conclusions

- The acceleration by using 2 GPU cards with comparison to single GPU cards are not as good as expected, but the acceleration by using 3 GPU cards with comparison to using 2 GPU cards are approximately linear.
    - I guess this is because, when using multiple GPU cards, to split the data, extra operators including `tf.split` and `tf.slice` are inserted into the network which cause extra memory copy. So the acceleration is not linear when using 2 GPU cards with comparison to 1 GPU cards. But to valid this need detail profiling.
- Data parallel will lead to a larger batch size, which requires more careful tuning for the hyper parameters, especially the optimization method, learning rate and the regularization rate, otherwise large batch size potentially harms the learning performance.

## Some potential further optimizations

1. I am not sure, whether NCCL `all-reduce` used in the lasted TensorFlow release can better help the gradient merge process.
1. The current data parallelism implementation is quite straight forward. `tf.split` and `tf.slice` are used to split the entire data batch into multiple input smaller mini-batch for each GPU cards. `tf.split` and `tf.slice` **will cause extra memory copy**, in essence, this canbe avoided.
    - The TensorFlow's Tensor2Tensor package gives a better implementation of data parallelism which uses the PS mode. I suppose it is the best practice suggested by TensorFlow internal developers. Maybe we can refer to it.
1. I do not modify the data feeding part. Currently, the workload is not balanced for each GPU cards. Let take an example:
    - Suppose 2 GPU cards are avilable.
    - The input bacth have 4 sequence paris and are padded to have the same length as the longest sequence in the batch. Suppose the actual lengths of 4 target sequence are: `[51, 34, 28, 26]`.
    - The original batch will be devided into 2 parts: `[51, 34]` and `[28, 26]`. In TensorFlow's dynamic rnn if actual sequence length is provided (this is the situation in the open sourced nmt package), the decoder computes the for the steps as many as the length of the longest sequence in the batch.
    - As a result, `/gpu:0` compute for 51 time steps while `/gpu:1` compute for 28 time steps.
    - After `/gpu:1` finished computations, it will wait for `/gpu:0` to finish the computation and then merge the gradients and calculate the parameter updates.