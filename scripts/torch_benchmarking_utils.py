from common_types import BiasType, GemmDataType

import torch


def rand_strided(size, stride, dtype=torch.float32, device="cpu", extra_size=0, zero=False):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(size, stride))
        + 1
        + extra_size
    )
    if dtype.is_floating_point and not zero:
        buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)

def gemm_dtype_to_torch_dtype(gemm_dtype):
    d = {
        GemmDataType.BF16: torch.bfloat16,
        GemmDataType.F16: torch.float16,
        GemmDataType.TF32: torch.float32,
    }
    return d[gemm_dtype]

def prep_torch_inputs(shape, gemm_dtype):
    b, m, n, k = shape.b, shape.m, shape.n, shape.k
    a_shape = [m, k]
    b_shape = [k, n]
    d_shape = [m, n]

    a_stride = [k, 1] if shape.layout.name.lower()[0] == "r" else [1, m]
    b_stride = [n, 1] if shape.layout.name.lower()[1] == "r" else [1, k]
    d_stride = [n, 1]

    if b > 1:
        for (arg_shape, arg_stride) in zip(
            [a_shape, b_shape, d_shape],
            [a_stride, b_stride, d_stride],
        ):
            arg_shape.insert(0, b)
            arg_stride.insert(0, arg_shape[-2] * arg_shape[-1])

    torch_dtype = gemm_dtype_to_torch_dtype(gemm_dtype)
    
    if shape.bias_type is not BiasType.NONE:
        c_shape = [n] if shape.bias_type.is_vector() else [m, n]
        c_stride = [1] if shape.bias_type.is_vector() else [n, 1]

        if shape.bias_type.is_batched():
            if len(c_shape) == 1:
                c_shape.insert(0, 1)
                c_stride.insert(0, n)

            c_shape.insert(0, b)
            c_stride.insert(0, c_shape[-2] * c_shape[-1])
        
        c_tensor = rand_strided(c_shape, c_stride, dtype=torch_dtype, device="cuda")

    a_tensor = rand_strided(a_shape, a_stride, dtype=torch_dtype, device="cuda")
    b_tensor = rand_strided(b_shape, b_stride, dtype=torch_dtype, device="cuda")
    d_tensor = rand_strided(d_shape, d_stride, dtype=torch_dtype, device="cuda", zero=True)

    # print(a_tensor.size(), a_tensor.stride())
    # print(b_tensor.size(), b_tensor.stride())
    # if shape.bias_type is not BiasType.NONE:
    #     print(c_tensor.size(), c_tensor.stride())
    # print(d_tensor.size(), d_tensor.stride())

    if shape.bias_type is BiasType.NONE:
        return a_tensor, b_tensor, d_tensor
    else:
        return a_tensor, b_tensor, c_tensor, d_tensor

def allclose(x, y, tol=1e-2):
    if x.dtype != y.dtype:
        raise RuntimeError(f'{x.dtype} did not match with {x.dtype}')
    if x.shape != y.shape:
        raise RuntimeError(f'{x.shape} did not match with {y.shape}')
    if x.dtype == torch.bool:
        return torch.sum(x ^ y) == 0
    if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        tol = 0
    diff = abs(x - y)
    x_max = torch.max(x)
    y_max = torch.max(y)
    err = torch.max(diff) / torch.max(x_max, y_max)
    return err <= tol

def do_bench(fn, warmup=25, rep=100, grad_to_none=None,
             percentiles=(0.5, 0.2, 0.8),
             record_clocks=False, fast_flush=False):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param percentiles: Performance percentile to return in addition to the median.
    :type percentiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    torch.cuda.synchronize()
    n_warmup = warmup
    n_repeat = rep
    
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    if percentiles:
        percentiles = torch.quantile(times, torch.tensor(percentiles)).tolist()
        return tuple(percentiles)
    else:
        return torch.mean(times).item()