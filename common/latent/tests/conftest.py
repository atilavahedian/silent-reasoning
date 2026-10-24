"""Test-session setup: one intra-op torch thread. The tests use tiny models, where multithreading gains nothing, and on
a shared, oversubscribed CPU (several agents' suites at once) the default thread pool made single tiny ops 100x
slower (layer_norm on a [1, 25, 32] tensor at about 14 ms). Numerics inside a test are unaffected: every comparison is
between computations of the same process."""
import torch

torch.set_num_threads(1)
