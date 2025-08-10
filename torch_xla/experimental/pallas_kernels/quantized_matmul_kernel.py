import functools
from typing import Any, Callable
import jax
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# --Start of Kernel--


def quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max: jax.Array,  # [1, bs_block_size]
):
  n_bits = 8
  int_max = 2**(n_bits - 1) - 1
  # TODO(kyuyeunk): Investigate performance gain from non xlu transpose.
  scale = jnp.transpose(x_abs_max / int_max)  # [bs_block_size, 1]
  x_int = jnp.round(x / scale).astype(jnp.int8)
  return x_int, scale


def get_vmem_limit(
    n_bs: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: jnp.dtype,
    w_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
    upper_limit_bytes: int = 96 * 1024 * 1024,
):
  # Calculate in/out VMEM size.
  x_size = batch_block_size * in_block_size * dtypes.bit_width(x_dtype)
  x_abs_max_size = batch_block_size * dtypes.bit_width(scale_dtype)
  w_size = out_block_size * in_block_size * dtypes.bit_width(w_dtype)
  scalar_size = out_block_size * dtypes.bit_width(scale_dtype)
  out_size = batch_block_size * out_block_size * dtypes.bit_width(out_dtype)

  vmem_in_out = x_size + x_abs_max_size + w_size + scalar_size + out_size
  vmem_in_out *= 2  # Account for compute and vreg spills.

  # Account for double buffering.
  # Double buffering is used only if there are multiple blocks per in/out.
  vmem_in_out += x_size if (n_bs > 1 or n_in > 1) else 0
  vmem_in_out += x_abs_max_size if (n_bs > 1) else 0
  vmem_in_out += w_size if (n_out > 1 or n_in > 1) else 0
  vmem_in_out += scalar_size if (n_out > 1) else 0
  vmem_in_out += out_size if (n_bs > 1 or n_out > 1) else 0

  # Calculate scratch VMEM size.
  acc_size = batch_block_size * out_block_size * dtypes.bit_width(acc_dtype)
  x_q_size = batch_block_size * in_block_size * dtypes.bit_width(x_q_dtype)
  x_scale_size = batch_block_size * dtypes.bit_width(scale_dtype)

  vmem_scratch = x_scale_size
  vmem_scratch += acc_size if save_acc else 0
  vmem_scratch += x_q_size if save_x_q else 0
  vmem_scratch *= 2  # Account for compute and vreg spills.

  # Add in/out and scratch VMEM size.
  vmem_used = vmem_in_out + vmem_scratch
  vmem_used_bytes = vmem_used // 8  # Convert bits to bytes.
  # Specify upper limit. Defaults to 96MB.
  vmem_limit_bytes = min(vmem_used_bytes, upper_limit_bytes)

  return vmem_limit_bytes


def validate_inputs(x, w_q, w_scale, x_abs_max, batch_block_size,
                    out_block_size, in_block_size):
  """Verify input shapes before invoking the kernel."""
  if x.shape[1] != w_q.shape[1]:
    raise ValueError(f'{x.shape[1]=} must be equal to {w_q.shape[1]=}')
  if w_q.shape[0] != w_scale.shape[1]:
    raise ValueError(f'{w_q.shape[0]=} must be equal to {w_scale.shape[1]=}')
  if x_abs_max.shape != (1, x.shape[0]):
    raise ValueError(f'{x_abs_max.shape=} must be equal to (1, {x.shape[0]=})')
  if x.shape[0] % batch_block_size != 0:
    raise ValueError(
        f'{x.shape[0]=} must be a multiple of block size {batch_block_size=}')
  if w_q.shape[0] % out_block_size != 0:
    raise ValueError(
        f'{w_q.shape[0]=} must be a multiple of block size {out_block_size=}')
  if x.shape[1] % in_block_size != 0:
    raise ValueError(
        f'{x.shape[1]=} must be a multiple of block size {in_block_size=}')


def matmul_kernel(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_q_ref: jax.Array,  # (out_block_size, in_block_size)
    w_scale_ref: jax.Array,  # (1, out_block_size)
    x_abs_max_ref: jax.Array,  # (1, batch_block_size)
    out_ref: jax.Array,  # (batch_block_size, out_block_size)
    acc_scratch: jax.Array,  # (batch_block_size, out_block_size)
    x_q_scratch: jax.Array,  # (batch_block_size, in_block_size)
    x_scale_scratch: jax.Array,  # (batch_block_size, 1)
    *,
    quantize_activation: bool,
    save_acc: bool,
    save_x_q: bool,
):
  out_idx, in_idx = pl.program_id(1), pl.program_id(2)
  n_in = pl.num_programs(2)
  x_ref_dtype = x_ref.dtype

  # Initialize conditional logic.
  if save_x_q:
    assert quantize_activation
    assert x_q_scratch is not None
    quant = (out_idx == 0)
  else:
    assert x_q_scratch is None
    quant = quantize_activation

  if save_acc:
    assert acc_scratch is not None
    is_first_step = (in_idx == 0)
    is_last_step = (in_idx == (n_in - 1))
  else:
    assert acc_scratch is None
    is_first_step = True
    is_last_step = True

  # Start of actual computation logic.
  def matmul_body(quant, is_first_step, is_last_step):
    if quantize_activation:
      if quant:
        if is_first_step:
          x_q, x_scale = quantize_array(x_ref[...], x_abs_max_ref[...])
          x_scale_scratch[...] = x_scale
        else:
          x_scale = x_scale_scratch[...]
          x_q = jnp.round(x_ref[...] / x_scale).astype(jnp.int8)

        if save_x_q:
          x_q_scratch[...] = x_q
      else:
        assert save_x_q
        x_q = x_q_scratch[...]
        if is_last_step:
          x_scale = x_scale_scratch[...]

      acc = jax.lax.dot_general(
          x_q,
          w_q_ref[...],
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.int32,
      )
    else:
      acc = jax.lax.dot_general(
          x_ref[...],
          w_q_ref[...],
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    if not is_first_step:
      acc += acc_scratch[...]

    if is_last_step:
      acc *= w_scale_ref[...]
      if quantize_activation:
        # TODO(kyuyeunk): Investigate performance gain from caching broadcast.
        acc *= x_scale
      out_ref[...] = acc.astype(x_ref_dtype)
    else:
      assert save_acc
      acc_scratch[...] = acc

  unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


@functools.partial(
    jax.jit,
    static_argnames=[
        'quantize_activation',
        'batch_block_size',
        'out_block_size',
        'in_block_size',
    ],
)
def quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_input_features]
    w_q: jax.Array,  # [n_output_features, n_input_features]
    w_scale: jax.Array,  # [n_output_features]
    zero_point: jax.Array | None = None,  # [n_output_features]
    quant_block_size: int | None = None,
    quantize_activation: bool = False,
    *,
    batch_block_size: int | None = None,
    out_block_size: int | None = None,
    in_block_size: int | None = None,
):
  assert zero_point is None, 'Not implemented: zero_point is not supported.'
  assert (quant_block_size
          is None), 'Not implemented: quant_block_size is not supported.'
  assert batch_block_size is not None and out_block_size is not None and in_block_size is not None

  # Pallas kernel only has access to a single block of the input. Therefere, for
  # per-token quantization, abs max has to be computed outside of the kernel.
  # TODO(kyuyeunk): Investigate performance gain from using Pallas for abs max.
  x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
  # Pallas requires minormost dim to be a multiple of sublane size 128.
  # Therefore, instead of using [bs, 1], we reshape this into [1, bs]
  x_abs_max = jnp.expand_dims(x_abs_max, axis=0)  # [1, bs]
  assert x_abs_max.shape == (1, x.shape[0])

  orig_bs, orig_in_features = x.shape
  orig_out_features, _ = w_q.shape

  # Pad the inputs to be multiple of block size.
  padded_bs = next_multiple(orig_bs, batch_block_size)
  if orig_bs < padded_bs:
    x = jnp.pad(x, ((0, padded_bs - orig_bs), (0, 0)))
    x_abs_max = jnp.pad(x_abs_max, ((0, 0), (0, padded_bs - orig_bs)))
  padded_out_features = next_multiple(orig_out_features, out_block_size)
  if orig_out_features < padded_out_features:
    w_q = jnp.pad(w_q, ((0, padded_out_features - orig_out_features), (0, 0)))
    w_scale = jnp.pad(w_scale, (0, padded_out_features - orig_out_features))
  padded_in_features = next_multiple(orig_in_features, in_block_size)
  if orig_in_features < padded_in_features:
    x = jnp.pad(x, ((0, 0), (0, padded_in_features - orig_in_features)))
    w_q = jnp.pad(w_q, ((0, 0), (0, padded_in_features - orig_in_features)))

  if w_scale.dtype != jnp.float32:
    w_scale = w_scale.astype(jnp.float32)
  w_scale = jnp.expand_dims(w_scale, axis=0)  # [1, n_output_features]

  n_bs = padded_bs // batch_block_size
  n_out = padded_out_features // out_block_size
  n_in = padded_in_features // in_block_size

  save_acc = n_in > 1
  # Remove redundant input quantization logic by caching quantized input. For
  # best performance, only enable this behavior when single input block is used
  # per batch.
  # TODO(kyuyeunk): Utilize customized pipeline for better performance.
  save_x_q = quantize_activation and n_in == 1 and n_out > 1

  acc_dtype = jnp.int32 if quantize_activation else jnp.float32

  vmem_limit_bytes = get_vmem_limit(
      n_bs=n_bs,
      n_out=n_out,
      n_in=n_in,
      batch_block_size=batch_block_size,
      out_block_size=out_block_size,
      in_block_size=in_block_size,
      x_dtype=x.dtype,
      w_dtype=w_q.dtype,
      x_q_dtype=jnp.int8,
      scale_dtype=jnp.float32,
      out_dtype=x.dtype,
      acc_dtype=acc_dtype,
      save_acc=save_acc,
      save_x_q=save_x_q,
  )

  kernel = pl.pallas_call(
      functools.partial(
          matmul_kernel,
          quantize_activation=quantize_activation,
          save_acc=save_acc,
          save_x_q=save_x_q,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i:
                           (b, i)),  # x
              pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i:
                           (o, i)),  # w_q
              pl.BlockSpec((1, out_block_size), lambda b, o, i:
                           (0, o)),  # w_scale
              pl.BlockSpec((1, batch_block_size), lambda b, o, i:
                           (0, b)),  # x_abs_max
          ],
          out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                 lambda b, o, i: (b, o)),
          scratch_shapes=[
              pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
              if save_acc else None,  # acc_scratch
              pltpu.VMEM((batch_block_size, in_block_size), jnp.int8)
              if save_x_q else None,  # x_q_scratch
              pltpu.VMEM((batch_block_size, 1),
                         jnp.bfloat16),  # x_scale_scratch
          ],
          grid=(n_bs, n_out, n_in),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out_features), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )

  validate_inputs(
      x,
      w_q,
      w_scale,
      x_abs_max,
      batch_block_size,
      out_block_size,
      in_block_size,
  )

  # The named_scope is used for autotune.
  kernel_name = get_kernel_name(batch_block_size, out_block_size, in_block_size)
  with jax.named_scope(kernel_name):
    out = kernel(x, w_q, w_scale, x_abs_max)

  return out[:orig_bs, :orig_out_features]


# --End of Kernel--

# --Start of Util--


def unfold_args(
    conditions: tuple[jax.Array | bool, ...],
    fn_conditions: tuple[bool, ...],
    fn: Callable[..., Any],
):
  """Minimize run-time branching of fn by converting jnp.bool to python bool."""
  if conditions:
    arg = conditions[0]
    if isinstance(arg, bool):
      unfold_args(conditions[1:], fn_conditions + (arg,), fn)
    else:
      assert arg.dtype == jnp.bool and arg.size == 1
      jax.lax.cond(
          arg,
          lambda: unfold_args(conditions[1:], fn_conditions + (True,), fn),
          lambda: unfold_args(conditions[1:], fn_conditions + (False,), fn),
      )
  else:
    fn(*fn_conditions)


def next_multiple(x, multiple):
  return ((x + multiple - 1) // multiple) * multiple


def get_kernel_name(bs_block_size, out_block_size, in_block_size):
  return f'quantized_matmul_kernel_{bs_block_size}_{out_block_size}_{in_block_size}'


# --End of util--

# --Start of tiles--

# Below are tuned block sizes.

# key:
#    - tpu_version
#    - batch_size
#    - n_output_features
#    - n_input_features
#    - activation_dtype
#    - quantize_activation
# value:
#    - batch_block_size
#    - out_block_size
#    - in_block_size
TUNED_BLOCK_SIZES = {
    # go/keep-sorted start
    (6, 1024, 1024, 4096, 'bfloat16', True): (1024, 256, 4096),
    (6, 1024, 1024, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 1024, 128, 8192, 'bfloat16', True): (512, 128, 8192),
    (6, 1024, 1280, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 1024, 13824, 5120, 'bfloat16', True): (1024, 768, 5120),
    (6, 1024, 14336, 4096, 'bfloat16', True): (1024, 1024, 4096),
    (6, 1024, 1792, 5120, 'bfloat16', True): (1024, 256, 5120),
    (6, 1024, 28672, 4096, 'bfloat16', True): (1024, 2048, 4096),
    (6, 1024, 3584, 18944, 'bfloat16', True): (1024, 3584, 512),
    (6, 1024, 3584, 3584, 'bfloat16', True): (1024, 512, 3584),
    (6, 1024, 3584, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 1024, 37888, 3584, 'bfloat16', True): (1024, 1024, 3584),
    (6, 1024, 4096, 14336, 'bfloat16', True): (1024, 256, 14336),
    (6, 1024, 4096, 4096, 'bfloat16', True): (1024, 512, 4096),
    (6, 1024, 4608, 3584, 'bfloat16', True): (1024, 512, 3584),
    (6, 1024, 5120, 1280, 'bfloat16', True): (1024, 1024, 1280),
    (6, 1024, 5120, 3456, 'bfloat16', True): (1024, 1024, 3456),
    (6, 1024, 5120, 640, 'bfloat16', True): (256, 5120, 640),
    (6, 1024, 5120, 6912, 'bfloat16', True): (1024, 1024, 6912),
    (6, 1024, 6144, 4096, 'bfloat16', True): (1024, 768, 4096),
    (6, 1024, 6912, 5120, 'bfloat16', True): (1024, 1152, 5120),
    (6, 1024, 7168, 8192, 'bfloat16', True): (1024, 512, 8192),
    (6, 1024, 8192, 1024, 'bfloat16', True): (1024, 4096, 1024),
    (6, 1024, 8192, 3584, 'bfloat16', True): (1024, 2048, 3584),
    (6, 1024, 896, 5120, 'bfloat16', True): (1024, 896, 2560),
    (6, 128, 1024, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 128, 1024, 8192, 'bfloat16', True): (128, 1024, 4096),
    (6, 128, 128, 8192, 'bfloat16', True): (128, 128, 8192),
    (6, 128, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 128, 13824, 5120, 'bfloat16', True): (128, 768, 5120),
    (6, 128, 14336, 4096, 'bfloat16', True): (128, 1024, 4096),
    (6, 128, 1792, 5120, 'bfloat16', True): (128, 1792, 1280),
    (6, 128, 28672, 4096, 'bfloat16', True): (128, 1024, 4096),
    (6, 128, 3584, 18944, 'bfloat16', True): (128, 256, 18944),
    (6, 128, 3584, 3584, 'bfloat16', True): (128, 896, 3584),
    (6, 128, 3584, 8192, 'bfloat16', True): (128, 512, 8192),
    (6, 128, 37888, 3584, 'bfloat16', True): (128, 18944, 256),
    (6, 128, 4096, 14336, 'bfloat16', True): (128, 256, 14336),
    (6, 128, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 128, 4608, 3584, 'bfloat16', True): (128, 1152, 3584),
    (6, 128, 5120, 1280, 'bfloat16', True): (128, 1280, 1280),
    (6, 128, 5120, 3456, 'bfloat16', True): (128, 1024, 3456),
    (6, 128, 5120, 640, 'bfloat16', True): (128, 2560, 640),
    (6, 128, 5120, 6912, 'bfloat16', True): (128, 512, 6912),
    (6, 128, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 128, 6912, 5120, 'bfloat16', True): (128, 3456, 1024),
    (6, 128, 7168, 8192, 'bfloat16', True): (128, 512, 8192),
    (6, 128, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 128, 8192, 3584, 'bfloat16', True): (128, 8192, 512),
    (6, 128, 896, 5120, 'bfloat16', True): (128, 896, 1280),
    (6, 16, 1024, 4096, 'bfloat16', True): (16, 1024, 2048),
    (6, 16, 1024, 8192, 'bfloat16', True): (16, 1024, 2048),
    (6, 16, 128, 8192, 'bfloat16', True): (16, 128, 4096),
    (6, 16, 1280, 8192, 'bfloat16', True): (16, 1280, 2048),
    (6, 16, 13824, 5120, 'bfloat16', True): (16, 768, 5120),
    (6, 16, 14336, 4096, 'bfloat16', True): (16, 1024, 4096),
    (6, 16, 1792, 5120, 'bfloat16', True): (16, 1792, 1280),
    (6, 16, 28672, 4096, 'bfloat16', True): (16, 1024, 4096),
    (6, 16, 3584, 18944, 'bfloat16', True): (16, 512, 18944),
    (6, 16, 3584, 3584, 'bfloat16', True): (16, 896, 3584),
    (6, 16, 3584, 8192, 'bfloat16', True): (16, 512, 8192),
    (6, 16, 37888, 3584, 'bfloat16', True): (16, 9472, 512),
    (6, 16, 4096, 14336, 'bfloat16', True): (16, 256, 14336),
    (6, 16, 4096, 4096, 'bfloat16', True): (16, 512, 4096),
    (6, 16, 4608, 3584, 'bfloat16', True): (16, 2304, 1792),
    (6, 16, 5120, 1280, 'bfloat16', True): (16, 1280, 1280),
    (6, 16, 5120, 3456, 'bfloat16', True): (16, 640, 3456),
    (6, 16, 5120, 640, 'bfloat16', True): (16, 2560, 640),
    (6, 16, 5120, 6912, 'bfloat16', True): (16, 512, 6912),
    (6, 16, 6144, 4096, 'bfloat16', True): (16, 768, 4096),
    (6, 16, 6912, 5120, 'bfloat16', True): (16, 768, 5120),
    (6, 16, 7168, 8192, 'bfloat16', True): (16, 1024, 4096),
    (6, 16, 8192, 1024, 'bfloat16', True): (16, 2048, 1024),
    (6, 16, 8192, 3584, 'bfloat16', True): (16, 8192, 512),
    (6, 16, 896, 5120, 'bfloat16', True): (16, 896, 2560),
    (6, 16384, 1024, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 16384, 1024, 8192, 'bfloat16', True): (512, 1024, 8192),
    (6, 16384, 128, 8192, 'bfloat16', True): (512, 128, 8192),
    (6, 16384, 1280, 8192, 'bfloat16', True): (2048, 1280, 8192),
    (6, 16384, 13824, 5120, 'bfloat16', True): (256, 2304, 5120),
    (6, 16384, 14336, 4096, 'bfloat16', True): (256, 3584, 4096),
    (6, 16384, 1792, 5120, 'bfloat16', True): (512, 1792, 5120),
    (6, 16384, 28672, 4096, 'bfloat16', True): (2048, 1792, 4096),
    (6, 16384, 3584, 18944, 'bfloat16', True): (2048, 3584, 512),
    (6, 16384, 3584, 3584, 'bfloat16', True): (512, 3584, 3584),
    (6, 16384, 3584, 8192, 'bfloat16', True): (1024, 3584, 8192),
    (6, 16384, 37888, 3584, 'bfloat16', True): (4096, 512, 3584),
    (6, 16384, 4096, 14336, 'bfloat16', True): (1024, 2048, 3584),
    (6, 16384, 4096, 4096, 'bfloat16', True): (1024, 4096, 4096),
    (6, 16384, 4608, 3584, 'bfloat16', True): (512, 4608, 3584),
    (6, 16384, 5120, 1280, 'bfloat16', True): (512, 5120, 1280),
    (6, 16384, 5120, 3456, 'bfloat16', True): (256, 5120, 3456),
    (6, 16384, 5120, 640, 'bfloat16', True): (512, 5120, 640),
    (6, 16384, 5120, 6912, 'bfloat16', True): (256, 5120, 6912),
    (6, 16384, 6144, 4096, 'bfloat16', True): (1024, 6144, 4096),
    (6, 16384, 6912, 5120, 'bfloat16', True): (512, 6912, 5120),
    (6, 16384, 7168, 8192, 'bfloat16', True): (256, 1792, 8192),
    (6, 16384, 8192, 1024, 'bfloat16', True): (512, 8192, 1024),
    (6, 16384, 8192, 3584, 'bfloat16', True): (2048, 2048, 3584),
    (6, 16384, 896, 5120, 'bfloat16', True): (512, 896, 5120),
    (6, 2048, 1024, 4096, 'bfloat16', True): (2048, 256, 4096),
    (6, 2048, 1024, 8192, 'bfloat16', True): (2048, 256, 8192),
    (6, 2048, 128, 8192, 'bfloat16', True): (512, 128, 8192),
    (6, 2048, 1280, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 2048, 13824, 5120, 'bfloat16', True): (2048, 768, 5120),
    (6, 2048, 14336, 4096, 'bfloat16', True): (2048, 1024, 4096),
    (6, 2048, 1792, 5120, 'bfloat16', True): (1024, 1792, 2560),
    (6, 2048, 28672, 4096, 'bfloat16', True): (2048, 1792, 4096),
    (6, 2048, 3584, 18944, 'bfloat16', True): (2048, 3584, 512),
    (6, 2048, 3584, 3584, 'bfloat16', True): (2048, 512, 3584),
    (6, 2048, 3584, 8192, 'bfloat16', True): (2048, 512, 8192),
    (6, 2048, 37888, 3584, 'bfloat16', True): (2048, 1024, 3584),
    (6, 2048, 4096, 14336, 'bfloat16', True): (2048, 4096, 512),
    (6, 2048, 4096, 4096, 'bfloat16', True): (2048, 512, 4096),
    (6, 2048, 4608, 3584, 'bfloat16', True): (1024, 768, 3584),
    (6, 2048, 5120, 1280, 'bfloat16', True): (256, 5120, 1280),
    (6, 2048, 5120, 3456, 'bfloat16', True): (2048, 512, 3456),
    (6, 2048, 5120, 640, 'bfloat16', True): (256, 5120, 640),
    (6, 2048, 5120, 6912, 'bfloat16', True): (1024, 1024, 6912),
    (6, 2048, 6144, 4096, 'bfloat16', True): (2048, 512, 4096),
    (6, 2048, 6912, 5120, 'bfloat16', True): (2048, 768, 5120),
    (6, 2048, 7168, 8192, 'bfloat16', True): (2048, 256, 8192),
    (6, 2048, 8192, 1024, 'bfloat16', True): (2048, 2048, 1024),
    (6, 2048, 8192, 3584, 'bfloat16', True): (2048, 1024, 3584),
    (6, 2048, 896, 5120, 'bfloat16', True): (512, 896, 5120),
    (6, 256, 1024, 4096, 'bfloat16', True): (256, 512, 4096),
    (6, 256, 1024, 8192, 'bfloat16', True): (256, 512, 8192),
    (6, 256, 128, 8192, 'bfloat16', True): (256, 128, 8192),
    (6, 256, 1280, 8192, 'bfloat16', True): (256, 1280, 2048),
    (6, 256, 13824, 5120, 'bfloat16', True): (256, 6912, 512),
    (6, 256, 14336, 4096, 'bfloat16', True): (256, 1024, 4096),
    (6, 256, 1792, 5120, 'bfloat16', True): (256, 896, 5120),
    (6, 256, 28672, 4096, 'bfloat16', True): (256, 896, 4096),
    (6, 256, 3584, 18944, 'bfloat16', True): (256, 512, 9472),
    (6, 256, 3584, 3584, 'bfloat16', True): (256, 3584, 896),
    (6, 256, 3584, 8192, 'bfloat16', True): (256, 3584, 1024),
    (6, 256, 37888, 3584, 'bfloat16', True): (256, 1024, 3584),
    (6, 256, 4096, 14336, 'bfloat16', True): (256, 4096, 1024),
    (6, 256, 4096, 4096, 'bfloat16', True): (256, 1024, 4096),
    (6, 256, 4608, 3584, 'bfloat16', True): (256, 1536, 3584),
    (6, 256, 5120, 1280, 'bfloat16', True): (256, 2560, 1280),
    (6, 256, 5120, 3456, 'bfloat16', True): (256, 1024, 3456),
    (6, 256, 5120, 640, 'bfloat16', True): (256, 5120, 640),
    (6, 256, 5120, 6912, 'bfloat16', True): (256, 512, 6912),
    (6, 256, 6144, 4096, 'bfloat16', True): (256, 768, 4096),
    (6, 256, 6912, 5120, 'bfloat16', True): (256, 768, 5120),
    (6, 256, 7168, 8192, 'bfloat16', True): (256, 3584, 2048),
    (6, 256, 8192, 1024, 'bfloat16', True): (256, 2048, 1024),
    (6, 256, 8192, 3584, 'bfloat16', True): (256, 1024, 3584),
    (6, 256, 896, 5120, 'bfloat16', True): (256, 896, 2560),
    (6, 32, 1024, 4096, 'bfloat16', True): (32, 512, 4096),
    (6, 32, 1024, 8192, 'bfloat16', True): (32, 256, 8192),
    (6, 32, 128, 8192, 'bfloat16', True): (32, 128, 8192),
    (6, 32, 1280, 8192, 'bfloat16', True): (32, 256, 8192),
    (6, 32, 13824, 5120, 'bfloat16', True): (32, 13824, 256),
    (6, 32, 14336, 4096, 'bfloat16', True): (32, 3584, 1024),
    (6, 32, 1792, 5120, 'bfloat16', True): (32, 896, 2560),
    (6, 32, 28672, 4096, 'bfloat16', True): (32, 7168, 512),
    (6, 32, 3584, 18944, 'bfloat16', True): (32, 896, 4736),
    (6, 32, 3584, 3584, 'bfloat16', True): (32, 896, 3584),
    (6, 32, 3584, 8192, 'bfloat16', True): (32, 512, 8192),
    (6, 32, 37888, 3584, 'bfloat16', True): (32, 1024, 3584),
    (6, 32, 4096, 14336, 'bfloat16', True): (32, 256, 14336),
    (6, 32, 4096, 4096, 'bfloat16', True): (32, 512, 4096),
    (6, 32, 4608, 3584, 'bfloat16', True): (32, 4608, 896),
    (6, 32, 5120, 1280, 'bfloat16', True): (32, 1280, 1280),
    (6, 32, 5120, 3456, 'bfloat16', True): (32, 1024, 3456),
    (6, 32, 5120, 640, 'bfloat16', True): (32, 2560, 640),
    (6, 32, 5120, 6912, 'bfloat16', True): (32, 512, 6912),
    (6, 32, 6144, 4096, 'bfloat16', True): (32, 1024, 4096),
    (6, 32, 6912, 5120, 'bfloat16', True): (32, 3456, 1280),
    (6, 32, 7168, 8192, 'bfloat16', True): (32, 512, 8192),
    (6, 32, 8192, 1024, 'bfloat16', True): (32, 2048, 1024),
    (6, 32, 8192, 3584, 'bfloat16', True): (32, 2048, 1792),
    (6, 32, 896, 5120, 'bfloat16', True): (32, 896, 2560),
    (6, 4096, 1024, 4096, 'bfloat16', True): (2048, 256, 4096),
    (6, 4096, 1024, 8192, 'bfloat16', True): (512, 1024, 8192),
    (6, 4096, 128, 8192, 'bfloat16', True): (2048, 128, 4096),
    (6, 4096, 1280, 8192, 'bfloat16', True): (256, 1280, 8192),
    (6, 4096, 13824, 5120, 'bfloat16', True): (2048, 256, 5120),
    (6, 4096, 14336, 4096, 'bfloat16', True): (4096, 512, 4096),
    (6, 4096, 1792, 5120, 'bfloat16', True): (2048, 256, 5120),
    (6, 4096, 28672, 4096, 'bfloat16', True): (4096, 512, 4096),
    (6, 4096, 3584, 18944, 'bfloat16', True): (1024, 3584, 512),
    (6, 4096, 3584, 3584, 'bfloat16', True): (4096, 512, 3584),
    (6, 4096, 3584, 8192, 'bfloat16', True): (1024, 1792, 8192),
    (6, 4096, 37888, 3584, 'bfloat16', True): (4096, 1024, 3584),
    (6, 4096, 4096, 14336, 'bfloat16', True): (2048, 4096, 512),
    (6, 4096, 4096, 4096, 'bfloat16', True): (4096, 512, 4096),
    (6, 4096, 4608, 3584, 'bfloat16', True): (2048, 768, 3584),
    (6, 4096, 5120, 1280, 'bfloat16', True): (256, 5120, 1280),
    (6, 4096, 5120, 3456, 'bfloat16', True): (4096, 512, 3456),
    (6, 4096, 5120, 640, 'bfloat16', True): (512, 5120, 640),
    (6, 4096, 5120, 6912, 'bfloat16', True): (256, 5120, 6912),
    (6, 4096, 6144, 4096, 'bfloat16', True): (2048, 768, 4096),
    (6, 4096, 6912, 5120, 'bfloat16', True): (1024, 2304, 5120),
    (6, 4096, 7168, 8192, 'bfloat16', True): (1024, 1792, 8192),
    (6, 4096, 8192, 1024, 'bfloat16', True): (256, 8192, 1024),
    (6, 4096, 8192, 3584, 'bfloat16', True): (2048, 1024, 3584),
    (6, 4096, 896, 5120, 'bfloat16', True): (512, 896, 5120),
    (6, 512, 1024, 4096, 'bfloat16', True): (512, 512, 4096),
    (6, 512, 1024, 8192, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 128, 8192, 'bfloat16', True): (512, 128, 8192),
    (6, 512, 1280, 8192, 'bfloat16', True): (512, 256, 8192),
    (6, 512, 13824, 5120, 'bfloat16', True): (512, 2304, 5120),
    (6, 512, 14336, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 1792, 5120, 'bfloat16', True): (512, 1792, 2560),
    (6, 512, 28672, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 3584, 18944, 'bfloat16', True): (512, 512, 18944),
    (6, 512, 3584, 3584, 'bfloat16', True): (512, 512, 3584),
    (6, 512, 3584, 8192, 'bfloat16', True): (512, 1792, 2048),
    (6, 512, 37888, 3584, 'bfloat16', True): (512, 4736, 3584),
    (6, 512, 4096, 14336, 'bfloat16', True): (512, 512, 14336),
    (6, 512, 4096, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 4608, 3584, 'bfloat16', True): (512, 768, 3584),
    (6, 512, 5120, 1280, 'bfloat16', True): (512, 2560, 1280),
    (6, 512, 5120, 3456, 'bfloat16', True): (512, 1280, 3456),
    (6, 512, 5120, 640, 'bfloat16', True): (256, 5120, 640),
    (6, 512, 5120, 6912, 'bfloat16', True): (512, 512, 6912),
    (6, 512, 6144, 4096, 'bfloat16', True): (512, 1536, 4096),
    (6, 512, 6912, 5120, 'bfloat16', True): (512, 1152, 5120),
    (6, 512, 7168, 8192, 'bfloat16', True): (512, 512, 8192),
    (6, 512, 8192, 1024, 'bfloat16', True): (512, 4096, 1024),
    (6, 512, 8192, 3584, 'bfloat16', True): (512, 1024, 3584),
    (6, 512, 896, 5120, 'bfloat16', True): (512, 896, 2560),
    (6, 64, 1024, 4096, 'bfloat16', True): (64, 512, 4096),
    (6, 64, 1024, 8192, 'bfloat16', True): (64, 512, 4096),
    (6, 64, 128, 8192, 'bfloat16', True): (64, 128, 8192),
    (6, 64, 1280, 8192, 'bfloat16', True): (64, 1280, 2048),
    (6, 64, 13824, 5120, 'bfloat16', True): (64, 768, 5120),
    (6, 64, 14336, 4096, 'bfloat16', True): (64, 896, 4096),
    (6, 64, 1792, 5120, 'bfloat16', True): (64, 1792, 1280),
    (6, 64, 28672, 4096, 'bfloat16', True): (64, 1024, 4096),
    (6, 64, 3584, 18944, 'bfloat16', True): (64, 896, 4736),
    (6, 64, 3584, 3584, 'bfloat16', True): (64, 3584, 896),
    (6, 64, 3584, 8192, 'bfloat16', True): (64, 1792, 2048),
    (6, 64, 37888, 3584, 'bfloat16', True): (64, 9472, 512),
    (6, 64, 4096, 14336, 'bfloat16', True): (64, 1024, 3584),
    (6, 64, 4096, 4096, 'bfloat16', True): (64, 4096, 1024),
    (6, 64, 4608, 3584, 'bfloat16', True): (64, 768, 3584),
    (6, 64, 5120, 1280, 'bfloat16', True): (64, 1280, 1280),
    (6, 64, 5120, 3456, 'bfloat16', True): (64, 1024, 3456),
    (6, 64, 5120, 640, 'bfloat16', True): (64, 2560, 640),
    (6, 64, 5120, 6912, 'bfloat16', True): (64, 5120, 768),
    (6, 64, 6144, 4096, 'bfloat16', True): (64, 768, 4096),
    (6, 64, 6912, 5120, 'bfloat16', True): (64, 1152, 5120),
    (6, 64, 7168, 8192, 'bfloat16', True): (64, 512, 8192),
    (6, 64, 8192, 1024, 'bfloat16', True): (64, 2048, 1024),
    (6, 64, 8192, 3584, 'bfloat16', True): (64, 1024, 3584),
    (6, 64, 896, 5120, 'bfloat16', True): (64, 896, 1280),
    (6, 8192, 1024, 4096, 'bfloat16', True): (1024, 1024, 4096),
    (6, 8192, 1024, 8192, 'bfloat16', True): (512, 1024, 8192),
    (6, 8192, 128, 8192, 'bfloat16', True): (256, 128, 8192),
    (6, 8192, 1280, 8192, 'bfloat16', True): (512, 1280, 8192),
    (6, 8192, 13824, 5120, 'bfloat16', True): (1024, 768, 5120),
    (6, 8192, 14336, 4096, 'bfloat16', True): (1024, 1792, 4096),
    (6, 8192, 1792, 5120, 'bfloat16', True): (1024, 1792, 5120),
    (6, 8192, 28672, 4096, 'bfloat16', True): (1024, 1792, 4096),
    (6, 8192, 3584, 18944, 'bfloat16', True): (4096, 1792, 512),
    (6, 8192, 3584, 3584, 'bfloat16', True): (2048, 512, 3584),
    (6, 8192, 3584, 8192, 'bfloat16', True): (512, 3584, 8192),
    (6, 8192, 37888, 3584, 'bfloat16', True): (2048, 512, 3584),
    (6, 8192, 4096, 14336, 'bfloat16', True): (1024, 2048, 3584),
    (6, 8192, 4096, 4096, 'bfloat16', True): (512, 4096, 4096),
    (6, 8192, 4608, 3584, 'bfloat16', True): (2048, 512, 3584),
    (6, 8192, 5120, 1280, 'bfloat16', True): (256, 5120, 1280),
    (6, 8192, 5120, 3456, 'bfloat16', True): (1024, 2560, 3456),
    (6, 8192, 5120, 640, 'bfloat16', True): (256, 5120, 640),
    (6, 8192, 5120, 6912, 'bfloat16', True): (512, 2560, 6912),
    (6, 8192, 6144, 4096, 'bfloat16', True): (1024, 3072, 4096),
    (6, 8192, 6912, 5120, 'bfloat16', True): (1024, 2304, 5120),
    (6, 8192, 7168, 8192, 'bfloat16', True): (256, 1792, 8192),
    (6, 8192, 8192, 1024, 'bfloat16', True): (256, 8192, 1024),
    (6, 8192, 8192, 3584, 'bfloat16', True): (1024, 8192, 3584),
    (6, 8192, 896, 5120, 'bfloat16', True): (1024, 896, 5120),
    # go/keep-sorted end
}


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[:-len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_key(
    batch_size,
    n_output_features,
    n_input_features,
    activation_dtype,
    quantize_activation,
):
  """Returns the key for the given parameters."""
  return (
      get_tpu_version(),
      batch_size,
      n_output_features,
      n_input_features,
      activation_dtype,
      quantize_activation,
  )


def get_tuned_block_sizes(block_table, batch_size, n_output_features,
                          n_input_features, activation_dtype,
                          quantize_activation):
  """
    Retrieve the tuned block sizes for the given parameters.

    Args:
        batch_size (int): The batch size.
        n_output_features (int): The number of output features.
        n_input_features (int): The number of input features.
        activation_dtype (str): The data type of the activation ('bfloat16' or 'float32').
        quantize_activation (bool): Whether to quantize the activation.

    Returns:
        tuple: A tuple containing the batch_block_size, out_block_size, and in_block_size.
  """
  key = get_key(
      batch_size,
      n_output_features,
      n_input_features,
      activation_dtype,
      quantize_activation,
  )
  return block_table.get(key, (None, None, None))


# --End of tiles--
