# Regent-FFT

[![Regent-FFT Tests](https://github.com/arjunkunna/regent-fft-arjun/actions/workflows/main.yml/badge.svg)](https://github.com/arjunkunna/regent-fft/actions/workflows/master.yml)

This is a fast fourier transform library built in Regent.

## Description

At a high level, the library takes the input matrix for the DFT in the form of a
region, and saves the output in an output region.

The library currently supports transforms up to 3 dimensions, and can be
configured to run on either a CPU or a GPU.

The CPU mode is supported by [FFTW](https://www.fftw.org/), and the GPU mode by
[cuFFT](https://developer.nvidia.com/cufft).

Both Complex-to-Complex and Real-To-Complex transformations are supported.

Both `complex64` and `complex32` types are supported in GPU mode. CPU mode is
only able to support `complex64`. It is possible to use `complex32` in CPU mode
but it requires some additional setup - please contact me if that is of
interest.

Batched transforms are also supported. 

## Getting Started

### Installing

First, make sure you have Regent installed. If you are using Sapling, please refer to the [Sapling guide](https://github.com/StanfordLegion/sapling-guide)). Note that you will want to load the following modules instead:

```shell
module load slurm mpi cmake cuda llvm
```

Otherwise, refer to the Regent installation instructions [here](https://regent-lang.org/install/) or the Legion ones [here](https://legion.stanford.edu/starting/)

Then, clone the repo:

```shell
https://github.com/arjunkunna/regent-fft.git
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

If operating in sapling, be sure to do the following on startup:

```shell
ssh <username>@sapling.stanford.edu
module load slurm mpi cmake cuda llvm
srun -n 1 -N 1 -c 40 -p gpu --exclusive --pty bash --login
<navigate to your .rg file>
source env.sh
../legion/language/regent.py test/fft_test.rg
```

## Usage

### Executing program

There are 3 possible modes:

1. GPU vs. CPU
2. Complex-to-Complex vs. Real-to-Complex
3. Float vs. Double (Float only supported in GPU mode)

API usage generally follows the following steps.

First, an FFT interface has to be generated depending on the type of transform
you hope to do. Then, we:

- create a plan,
- execute the said plan, and then
- destroy the plan once we are done.

There are several sample code snippets in the `fft_test.rg` file for reference
as well.

#### 1. Link the `fft.rg` file and generate an interface

- The first argument is the dimension - `int1d`, `int2d`, or `int3d`.
- The second argument is the data type of the input - `complex64`, `complex32`,
  `real`, or `double`.
- The third argument is the data type of the input - `complex64` or `complex32`.

```lua
local fft = require("fft")
local fft1d = fft.generate_fft_interface(int1d, complex64, complex64)
```

#### 2. Make a plan

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
transform with doubles, the input array will have fieldspace `double` and output
array will have fieldspace `complex64`.

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
to execute it in a separate task, they must wrap the task themselves

#### 3. Execute the plan

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

#### 4. Destroy the plan

When a plan is no longer needed, it can be destroyed:

```lua
fft1d.destroy_plan(p)
```

### 5. Batched Transforms

To illustrate how to perform a batched transform, let us use the example where you want to perform 7 batches of a 256 x 256 transform. 

In this case, the user creates a 3D interface:

```
local fft3d_batch = fft.generate_fft_interface(int3d, complex64, complex64)
```

The input and ouput regions should be 256 x 256 x 7 arrays: i.e., the last dimension is the number of batches. The plan region remains the same as before:

```
var r = region(ispace(int3d, {256, 256, 7}), complex64)
var s = region(ispace(int3d, {256, 256, 7}), complex64)
var p = region(ispace(int1d, 1), fft3d_batch_real.plan)
```

The key difference is that we call `make_plan_batch` instead of `make_plan`

```
fft3d_batch_real.make_plan_batch(r, s, p)
```

Then, we execute and destroy as in the regular case.

```
fft3d_batch_real.execute_plan_task(r, s, p)
fft3d_batch_real.destroy_plan(p)
```

As you can see, the main points of differentiation from the regular transform API is that we input a region of dimension `n+1`, where the final dimension is the number of batches, and use `make_plan_batch` instead of `make_plan`

Please also refer to the `test3d_batch` and `test3d_batch_real` examples in `fft_test.rg` for reference.

Batched transforms are supported on both CPU and GPUs, for 1 and 2 dimensions. For GPUs, both real-to-complex and complex-to-complex tranforms are supported (for both `complex32` and `complex64`). For CPU, only `complex64`-to-`complex64` transforms are supported currently.

## Future Developments

Next items in the pipeline include batch transforms, as well as distributed
transforms across multiple nodes. Please let us know if there are specific
features that may be helpful.

## Authors

- Elliott Slaughter (<slaughter@cs.stanford.edu>)
- Arjun Kunna (<arjunkunna@gmail.com>)

## Version History

- 1.0
  - Initial Release - Supports CPU and single-GPU transforms for 1D, 2D, and 3D.
  - Supports Real-to-Complex and Complex-to-Complex transforms for both CPU (complex64 only) and GPU (complex32 and complex64).
  - Supports batched transforms: both R2C and C2C (complex32 and complex64) for GPUs, and R2C and C2C (only complex64) for CPUs.

## Additional Resources

- For information on Regent, please refer to [the Regent
  website](https://regent-lang.org/) or [the Legion website](https://legion.stanford.edu/starting/)

- For information on the FFT transform, please refer to these set of
  [notes](https://web.stanford.edu/class/cs168/l/l15.pdf) or this
  [course](https://see.stanford.edu/Course/EE261)

## Acknowledgments

- [FFTW](https://www.fftw.org/)
- [cuFFT](https://developer.nvidia.com/cufft)
- [Regent](https://regent-lang.org/)
- [Legion](https://legion.stanford.edu/starting/)
