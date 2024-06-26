# Regent-FFT

[![Regent-FFT
Tests](https://github.com/StanfordLegion/regent-fft/actions/workflows/main.yml/badge.svg)](https://github.com/StanfordLegion/regent-fft/actions)

This is a fast fourier transform (FFT) library built in Regent.

## Description

At a high level, the library takes the input matrix for the Discrete Fourier
Transform in the form of a region, and saves the output in an output region.

The library currently supports transforms up to 3 dimensions, and can be
configured to run on either a CPU or a GPU.

The CPU mode is powered by [FFTW](https://www.fftw.org/), and the GPU mode by
[cuFFT](https://developer.nvidia.com/cufft).

Both Complex-to-Complex and Real-To-Complex transformations are supported.

Both single-precision and double-precision modes are supported i.e. both `float`
/ `complex32` and `double` / `complex64` types.

Batched transforms are also supported.

## Getting Started

### Install

First, make sure you have Regent installed. (If you are using Sapling, please
refer to the [Sapling guide](https://github.com/StanfordLegion/sapling-guide)).
Be sure to install with `MAX_DIM=4` flag enabled.

Note that you will want to load the following modules instead:

```shell
module load slurm mpi cmake cuda llvm
```

Otherwise, refer to the Regent installation instructions
[here](https://regent-lang.org/install/) or the Legion ones
[here](https://legion.stanford.edu/starting/).

Then, clone the repo:

```shell
git clone https://github.com/StanfordLegion/regent-fft.git
```

Next, run the install script and add environment variables:

```shell
./install.py
source env.sh
```

Then, run your `.rg` script, which can be set up using the instructions in the
'Executing Program' section:

```shell
../legion/language/regent.py test/fft_test.rg
```

If operating in sapling, the following flow will likely be typical for
subsequent usages:

```shell
ssh <username>@sapling.stanford.edu
module load slurm mpi cmake cuda llvm
srun -n 1 -N 1 -c 40 -p gpu --exclusive --pty bash --login
<navigate to your .rg file>
source env.sh
../legion/language/regent.py test/fft_test.rg
```

## Usage

### Execute Program

There are 4 possible modes:

1. GPU vs. CPU
2. Complex-to-Complex vs. Real-to-Complex
3. Float vs. Double
4. Batched mode (supported for all of the above)

API usage generally follows the following steps.

First, an FFT interface has to be generated depending on the type of transform
you hope to do. Then, we:

- Create a plan,
- Execute the said plan, and then
- Destroy the plan once we are done.

There are several sample code snippets in the `fft_test.rg` file for reference
as well.

#### 1. Import `fft.rg`

- The first argument is the dimension: `int1d`, `int2d`, or `int3d`.
- The second argument is the data type of the input: `complex64`, `complex32`,
  `float`, or `double`.
- The third argument is the data type of the output: `complex64` or `complex32`.

```lua
local fft = require("fft")
local fft1d = fft.generate_fft_interface(int1d, complex64, complex64)
```

#### 2. Make Plan

`make_plan` takes three arguments:

1. `r`: input region
2. `s`: output region
3. `p`: plan region

```lua
fft1d.make_plan(r, s, p)
```

The input region should be initialized with index space of the form
`ispace(<type>, N)`, where N is the size of the array, and `<type>` is either
int1d/int2d/int3d depending on the dimension of the transform. The fieldspace of
the region is the type supported by the transform - e.g, in a real-to-complex
transform with double precision, the input array will have fieldspace `double`
and output array will have fieldspace `complex64`. Larger fieldspaces that
contain the appropriate types can also be passed in via field polymorphism -
[here](https://groups.google.com/g/legionusers/c/yvQa8BE6QD0/m/_1cL_w-aAAAJ) is
an example.

For example, in a 1D double-to-complex64 transform of size 3, the input and
output regions may be initialized as follows:

```lua
var r = region(ispace(int1d, 3), double)
var s = region(ispace(int1d, 3), complex64)
```

The plan region always takes the following form, with fieldspace `fft.plan` (see
`fft.rg` for description of plan fieldspace):

```lua
var p = region(ispace(int1d, 1), fft1d.plan)
```

`make_plan` is a `__demand(__inline)` task. This means that if the user wants it
to execute it in a separate task, they must wrap the task themselves.

> [!IMPORTANT]
>
> `make_plan` overwrites the input and output regions `r` and `s`. This is
> mandated by FFTW, which Regent FFT uses on CPUs. In order to avoid overwriting
> data, the user must either initialize the plan prior to loading the regions
> with data, or else must create a temporary region (of the same size and layout
> as the real one) for use in initialization.

#### 3. Execute Plan

Next, we execute the plan. This takes the same 3 regions as mentioned above.

```lua
fft1d.execute_plan_task(r, s, p)
```

Note that `execute_plan` is a `__demand(__inline)` task (similar to `make_plan`
above). The task `execute_plan_task` is simply a wrapper around `execute_plan`
for convenience, to avoid needing to define this explicitly.

> [!IMPORTANT]
>
> Because `execute_plan` is a `__demand(__inline)` task, it will never execute
> on the GPU (unless the parent task is running on the GPU). Therefore, in most
> cases, it is necessary to use `execute_plan_task` if one wants to use the GPU.

#### 4. Destroy Plan

When a plan is no longer needed, it can be destroyed:

```lua
fft1d.destroy_plan(p)
```

#### 5. Batched Transforms

To illustrate how to perform a batched transform, let us use the example where
you want to perform 7 batches of a 256 x 256 transform.

Since the transform is a 2D one, the user creates a interface with `itype` of
dimension N+1: in this case, an `int3d`. The last dimension is used to store the
number of batches.

```lua
local fft2d_batch_complex64_complex64 = fft.generate_fft_interface(int3d, complex64, complex64)
```

The input and output regions should be 256 x 256 x 7 arrays: i.e., the last
dimension is the number of batches. The plan region remains the same as before:

```lua
var r = region(ispace(int3d, {256, 256, 7}), complex64)
var s = region(ispace(int3d, {256, 256, 7}), complex64)
var p = region(ispace(int1d, 1), fft3d_batch_real.plan)
```

The key difference is that we call `make_plan_batch` instead of `make_plan`

```lua
fft2d_batch_complex64_complex64.make_plan_batch(r, s, p)
```

Then, we execute and destroy as in the regular case.

```lua
fft2d_batch_complex64_complex64.execute_plan_task(r, s, p)
fft2d_batch_complex64_complex64.destroy_plan(p)
```

As you can see, the main points of differentiation from the regular transform
API is that we input a region of dimension `n+1`, where the final dimension is
the number of batches, and use `make_plan_batch` instead of `make_plan`

Please also refer to the `test_2d_complex64_to_complex64_batch_transform` and
`test_2d_double_to_complex64_batch_transform` examples in `fft_test.rg` for
reference.

Batched transforms are supported on both CPU and GPUs, for 1, 2 and 3
dimensions. Both single-precision and double-precision real-to-complex and
complex-to-complex transforms are supported.

#### 6. Distributed Mode

The API also supports a distributed mode, where every machine in a distributed
job executes an independent FFT of the same size.

To initialize in distributed mode, we might do the following (we have `n` 1-D
`complex64` to `complex64` transforms of size 3):

```lua
var n = fft1d.get_num_nodes()
var p = region(ispace(int1d, n), fft1d.plan)
var p_part = partition(equal, p, ispace(int1d, n))
var r = region(ispace(int1d, 3*n), complex64)
var r_part = partition(equal, r, ispace(int1d, n))
var s = region(ispace(int1d, 3*n), complex64)
var s_part = partition(equal, s, ispace(int1d, n))
fft1d.make_plan_distrib(r, r_part, s, s_part, p, p_part)
```

Note the use of `get_num_nodes` to determine the size of the `p` region and
partition. The task `make_plan_distrib` is a `__demand(__inline)` task that
internally performs an index launch over the machine to initialize `p`.

> [!IMPORTANT]
>
> Like `make_plan`, `make_plan_distrib` overwrites the input and output regions
> `r` and `s`.

> [!IMPORTANT]
>
> In order for the distributed API to work correctly, it is essential that each
> task in the index launch inside of `make_plan_distrib` is mapped onto a
> separate node. This ensures that when the region `p` is used later, there is a
> plan for every node in the machine.

If there are partitions `r_part` and `s_part` that are distributed around the
machine, one might do the following to execute the plan:

```lua
__demand(__index_launch)
for i in r_part.colors do
  fft1d.execute_plan_task(r_part[i], s_part[i], p)
end
```

Note: `execute_plan` is a `__demand(__inline)` task (as described above). The
task `execute_plan_task` is simply a wrapper around execute_plan for
convenience, to avoid needing to define this explicitly.

> [!IMPORTANT]
>
> Because execute_plan is a `__demand(__inline)` task, it will never execute on
> the GPU (unless the parent task is running on the GPU). Therefore, in most
> cases it is necessary to use `execute_plan_task` if one wants to use the GPU.

> [!IMPORTANT]
>
> While `execute_plan_task` may be executed on the GPU, the contents of the `p`
> region must still be available on the CPU, because the plans must be used by
> the host-side code to launch the FFT kernels. Therefore, when
> `execute_plan_task` is mapped onto the GPU, it is very important to map the
> `p` region into zero-copy memory.

Lastly, to destroy the plan:

```lua
fft1d.destroy_plan_distrib(p, p_part)
```

> [!NOTE]
>
> This is a `__demand(__inline)` task, and `destroy_plan_distrib` will
> internally perform an index launch to destroy the plans on each node.

> [!IMPORTANT]
>
> like `make_plan_distrib`, the index launch issued by `destroy_plan_distrib`
> must be mapped so that each point task runs on the node where the plan was
> originally created.

## Future Developments

Next items in the pipeline include distributed transforms across multiple nodes.
Please submit an issue if there are specific features that may be helpful.

## Authors

- Elliott Slaughter (<slaughter@cs.stanford.edu>)
- Arjun Kunna (<arjunkunna@gmail.com>)

## Additional Resources

- For information on Regent, please refer to the [Regent
  website](https://regent-lang.org/) or the [Legion
  website](https://legion.stanford.edu/starting/).

- For information on the FFT transform, please refer to these set of
  [notes](https://web.stanford.edu/class/cs168/l/l15.pdf) or this
  [course](https://see.stanford.edu/Course/EE261).

## Acknowledgments

- [FFTW](https://www.fftw.org/)
- [cuFFT](https://developer.nvidia.com/cufft)
- [Regent](https://regent-lang.org/)
- [Legion](https://legion.stanford.edu/starting/)
