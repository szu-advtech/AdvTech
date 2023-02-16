# EPCC OpenMP micro-benchmark suite
A fork of the EPCC OpenMP micro-benchmark suite with some improvements

If you just want the original, you can download it here:
https://www.epcc.ed.ac.uk/research/computing/performance-characterisation-and-benchmarking/epcc-openmp-micro-benchmark-suite

## Some notes on parameter tuning for `schedbench`
Exactly what is being measured is a function of the parameters. Bad parameters can result in measuring the wrong thing. If you are unsure whether you have isolated scheduling overheads, verify that your results look similar to the results in [1]. In particular, you should check that the overheads for the `dynamic` schedule decreases as the chunk size goes from 1, to 2, to 4.

Here are some pointers to achieve this:

#### Avoid a low iteration count
A low iteration count in the parallel loop can lead to measuring parallel for-loop overhead.

#### Avoid a long delay time
A long delay time can make the inner loops take a while. The time spent on scheduling is orders of magnitude less. Because the benchmark measures overhead as `avg_test_time - avg_reference_time`, this leads to the scheduling overheads getting "drowned out" by the time spent waiting.

#### Use one thread per physical core
If your system supports simultaneous multithreading (e.g. hyper-threading), you risk having logical threads competing for resources. In my experiments, the parallel tests took 40% longer than the reference test when using 8 threads on my quad-core Intel Core i7-2600K.

#### Example parameters
The following parameters gave good results on my system:

- `itersperthr = 8192` (must be changed in `schedbench.c`)
- `--outer-repetitions 50` (recommended in epcc paper[1])
- `--delay-time 0.01`
- `--test-time 2000`

# References
- [1] Bull, J. Mark. "Measuring synchronisation and scheduling overheads in OpenMP." *Proceedings of First European Workshop on OpenMP*. Vol. 8. 1999.
