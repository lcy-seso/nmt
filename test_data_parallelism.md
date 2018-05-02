# Test data parallelism for NMT model

## codes

- Use this codes: https://github.com/lcy-seso/nmt/compare/format_and_add_precommit...lcy-seso:add_data_parallelism
  - commit : https://github.com/lcy-seso/nmt/commit/2d73467680c9944b0f90b6df52efb580476c81ef

## Settings

1. Disable dataset shuffle, so that running the experiment for multiple times to run multiple batches will use the same training data and the data are feeded to the network in the same order.
1. Run 50 mini-batches, and calculate the time that how many samples processed every second.

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
