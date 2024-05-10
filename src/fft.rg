-- Copyright 2024 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Regent FFT library

import "regent"

local format = require("std/format")
local launcher = require("std/launcher")
local data = require("common/data")
local gpuhelper = require("regent/gpu/helper")
local gpu_available = gpuhelper.check_gpu_available()

-- Import C and FFTW APIs
local c = regentlib.c
local fftw_c = terralib.includec("fftw3.h")
regentlib.linklibrary("libfftw3.so")

-- Import cuFFT API
local cufft_c
if gpu_available then
  cufft_c = terralib.includec("cufftXt.h")
  terralib.linklibrary("libcufft.so")
end

-- Define constants
fftw_c.FFTW_FORWARD = -1
fftw_c.FFTW_BACKWARD = 1
fftw_c.FFTW_MEASURE = 0
fftw_c.FFTW_ESTIMATE = (2 ^ 6)

local fft = {}

--- Create the FFT interface.
-- @param itype Index type of transform (int2d/int2d/int3d).
-- @param dtype_in Input data type of transform (float/double/complex32/complex64).
-- @param dtype_out Output data type of transform (complex32/complex64).
function fft.generate_fft_interface(itype, dtype_in, dtype_out)
  assert(regentlib.is_index_type(itype), "requires an index type as the first argument")
  local dim = itype.dim
  local dtype_size = terralib.sizeof(dtype_out)

  -- Identify if it is a R2C transform: real_flag is true if an R2C transform
  local real_flag = false
  if dtype_in == double or dtype_in == float then
    real_flag = true
  end

  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")

  local iface = {}

  -- Create fspaces depending on whether GPUs are used or not
  local iface_plan
  if gpu_available then
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      float_p : fftw_c.fftwf_plan,
      cufft_p : cufft_c.cufftHandle,
      address_space : c.legion_address_space_t,
    }
  else
    fspace iface_plan {
      p : fftw_c.fftw_plan,
      float_p : fftw_c.fftwf_plan,
      address_space : c.legion_address_space_t,
    }
  end

  iface.plan = iface_plan
  iface.plan.__no_field_slicing = true

  -- Create function that returns a base_pointer to a region with ispace of
  -- dimension d and fspace t d is dimension, t is dtype/region fspace
  -- (complex64)
  local function make_get_base(d, t)
    local rect_t = c["legion_rect_" .. d .. "d_t"]
    local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. d .. "d"]
    local raw_rect_ptr = c["legion_accessor_array_" .. d .. "d_raw_rect_ptr"]
    local destroy_accessor = c["legion_accessor_array_" .. d .. "d_destroy"]

    local terra get_base(rect : rect_t, physical : c.legion_physical_region_t, field : c.legion_field_id_t)
      var subrect : rect_t
      var offsets : c.legion_byte_offset_t[d]
      var accessor = get_accessor(physical, field)
      var base_pointer = [&t](raw_rect_ptr(accessor, rect, &subrect, &(offsets[0])))
      regentlib.assert(base_pointer ~= nil, "base pointer is nil")
      escape
        for i = 0, d-1 do
          emit quote
            regentlib.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
          end
        end
      end

      -- regentlib.assert(offsets[0].offset == terralib.sizeof(complex64), "stride does not match expected value")
      destroy_accessor(accessor)
      return base_pointer
    end

    -- Function to get base pointer of region: returns base_pointer
    local terra get_offset(rect : rect_t, physical : c.legion_physical_region_t, field : c.legion_field_id_t)
      var subrect : rect_t
      var offsets : c.legion_byte_offset_t[d]
      var accessor = get_accessor(physical, field)
      var base_pointer = [&t](raw_rect_ptr(accessor, rect, &subrect, &(offsets[0])))

      -- regentlib.assert(offsets[0].offset == terralib.sizeof(complex64), "stride does not match expected value")
      destroy_accessor(accessor)
      return offsets
    end

    return rect_t, get_base, get_offset
  end

  -- Define get_base functions for input, output, and plan regions
  local rect_plan_t, get_base_plan, get_offset_plan = make_get_base(1, iface.plan) --get_base_plan returns a base_pointer to a region with fspace iface.plan. (always dim = 1 because plan regions are dim 1: 'var p = region(ispace(int1d, 1), fft1d.plan)')
  local rect_in_t, get_base_in, get_offset_in = make_get_base(dim, dtype_in) --get_base returns a base pointer to a region with fspace dtype
  local rect_out_t, get_base_out, get_offset_out = make_get_base(dim, dtype_out) --get_base returns a base pointer to a region with fspace dtype

  -- Takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
  -- Used to branch on CPU vs GPU execution inside tasks
  local terra get_executing_processor(runtime : c.legion_runtime_t)
    var ctx = c.legion_runtime_get_context()
    var result = c.legion_runtime_get_executing_processor(runtime, ctx)
    c.legion_context_destroy(ctx)
    return result
  end

  -- NOTE: Keep this in sync with default_mapper.h
  local DEFAULT_TUNABLE_NODE_COUNT = 0
  local DEFAULT_TUNABLE_LOCAL_CPUS = 1
  local DEFAULT_TUNABLE_LOCAL_GPUS = 2
  local DEFAULT_TUNABLE_LOCAL_IOS = 3
  local DEFAULT_TUNABLE_LOCAL_OMPS = 4
  local DEFAULT_TUNABLE_LOCAL_PYS = 5
  local DEFAULT_TUNABLE_GLOBAL_CPUS = 6
  local DEFAULT_TUNABLE_GLOBAL_GPUS = 7
  local DEFAULT_TUNABLE_GLOBAL_IOS = 8
  local DEFAULT_TUNABLE_GLOBAL_OMPS = 9
  local DEFAULT_TUNABLE_GLOBAL_PYS = 10

  __demand(__inline)
  task iface.get_tunable(tunable_id : int)
    var f = c.legion_runtime_select_tunable_value(__runtime(), __context(), tunable_id, 0, 0)
    var n = __future(int64, f)
    return n
  end

  __demand(__inline)
  task iface.get_num_nodes()
    return iface.get_tunable(DEFAULT_TUNABLE_NODE_COUNT)
  end

  __demand(__inline)
  task iface.get_num_local_gpus()
    return iface.get_tunable(DEFAULT_TUNABLE_LOCAL_GPUS)
  end

   __demand(__inline)
  task iface.get_plan(plan : region(ispace(int1d), iface.plan), check : bool) : &iface.plan
  where reads(plan) do
    format.println("In get_plan...")

    -- Get physical region __physical(r.{f, g, ...}) returns an array of
    -- physical regions (legion_physical_region_t) for r, one per field, for
    -- fields f, g, etc. in the order that the fields are listed in the call.
    var pr = __physical(plan)[0] --returns first physical region

    regentlib.assert(c.legion_physical_region_get_memory_count(pr) == 1, "plan instance has more than one memory?")

    -- Ensure that plan is in the right kind of memory
    var mem_kind = c.legion_memory_kind(c.legion_physical_region_get_memory(pr, 0))
    regentlib.assert(mem_kind == c.SYSTEM_MEM or mem_kind == c.REGDMA_MEM or mem_kind == c.Z_COPY_MEM, "plan instance must be stored in sysmem, regmem, or zero copy mem")

    -- Get pointer to plan: get_base_plan returns a base_pointer to a region with fspace iface.plan
    format.println("Getting plan_base...")
    var plan_base = get_base_plan(rect_plan_t(plan.ispace.bounds), __physical(plan)[0], __fields(plan)[0])
    var i = c.legion_processor_address_space(get_executing_processor(__runtime()))

    var p : &iface.plan
    var bounds = plan.ispace.bounds

    -- T(x) is a cast from type T to a value x: int1d(1) = number 1 (casted to type int1d)
    if bounds.hi - bounds.lo + 1 > int1d(1) then
      p = plan_base + i
    else
      p = plan_base
    end

    regentlib.assert(not check or p.address_space == i, "plans can only be used on the node where they are originally created")
    format.println("Returning plan_base...")
    return p
  end

  -- MAKE PLAN FUNCTIONS

  -- Functions to create plan. Takes input, output, and plan regions and makes
  -- plan using cufft/FFTW functions, storing it in iface_plan.p

  -- Task: Make plan in GPU version. Calls cufftPlanMany and stores plan in cufft_p
  local make_plan_gpu
  rescape
    if gpu_available then
      remit rquote
        __demand(__cuda, __leaf)
        task make_plan_gpu(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan), address_space : c.legion_address_space_t)
        where reads writes(input, output, plan) do
          format.println("In iface.make_plan_gpu...")

          -- Get pointer to plan
          var p = iface.get_plan(plan, true)

          -- Verify we are in GPU mode by checking TOC_PROC Takes a
          -- c.legion_runtime_t and returns
          -- c.legion_runtime_get_executing_processor(runtime, ctx)
          var proc = get_executing_processor(__runtime())
          format.println("Make_Plan_GPU: TOC PROC IS {}",c.TOC_PROC)
          format.println("Make_Plan_GPU: Processor kind is {}", c.legion_processor_kind(proc))

          if c.legion_processor_kind(proc) == c.TOC_PROC then
            format.println("Processor is TOC, so running GPU functions")
            var i = c.legion_processor_address_space(proc)
            regentlib.assert(address_space == i, "make_plan_gpu must be executed on a processor in the same address space")

            -- Get input and output bases
            var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
            var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
            var lo = input.ispace.bounds.lo:to_point()
            var hi = input.ispace.bounds.hi:to_point()
            var n : int[dim] --n is an array of size dim with the size of each dimension in the entries
            ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

            -- Call cufftPlanMany: cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch) --rank = dimensionality of transform (1,2,3)
            format.println("Calling cufftPlanMany...")

            var ok = 0
            if dtype_size == 8 and real_flag then
              format.println("Calling cufftPlanMany with CUFFT_R2C: Float to Complex32 ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_R2C, 1)
            elseif dtype_size == 8 then
              format.println("Calling cufftPlanMany with CUFFT_C2C: Complex32 to Complex32 ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_C2C, 1)
            elseif real_flag and dtype_size == 16 then
              format.println("Calling cufftPlanMany with CUFFT_D2Z: Complex32 to Complex32 ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_D2Z, 1)
            elseif dtype_size == 16 then
              format.println("Calling cufftPlanMany with CUFFT_Z2Z: Complex64 to Complex64 ... ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_Z2Z, 1)
            end

            -- Check return value of cufftPlanMany
            if ok == cufft_c.CUFFT_INVALID_VALUE then
              format.println("Invalid value in cufftPlanMany")
            end

            regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftPlanMany failed")
            format.println("cufftPlanMany Successful")

          -- GPU not identified: return error
          else
            regentlib.assert(false, "make_plan_gpu must be executed on a GPU processor")
          end
        end
      end
    end
  end

  --- Make plan.
  -- @param input Input region.
  -- @param output Output region.
  -- @param plan Plan region.
  -- @note To execute it in a separate task, it must be wrapped into a task.
  -- @note Calls `make_plan_gpu` if necessary.
   __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), dtype_in),output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    format.println("In iface.make_plan...")

    -- Obtain pointer to plan using get_plan
    format.println("Calling get_plan...")
    var p = iface.get_plan(plan, false)

    -- Get_executing process: takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime())) --legion_processor_address_space: takes a legion_processor_t proc and returns a legion_address_space_t

    -- Check input/output bounds. Bounds code example: var is = ispace(int1d, 12, -1). is.bounds returns rect1d { lo = int1d(-1), hi = int1d(10) }
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")

    -- Get pointers to input and output regions
    -- get_base returns a base pointer to a region with ispace of dimension dim and fspace dtype. [local rect_t, get_base = make_get_base(dim, dtype)] [local terra get_base(rect : rect_t, physical : c.legion_physical_region_t, field : c.legion_field_id_t)]

    var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    -- Info on argumentes from https://legion.stanford.edu/doxygen/class_legion_1_1_physical_region.html
    -- __physical(r.{f, g, ...}) returns an array of physical regions (legion_physical_region_t) for r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.
    -- __fields(r.{f, g, ...}) returns an array of the field IDs (legion_field_id_t) of r, one per field, for fields f, g, etc. in the order that the fields are listed in the call.

    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()

    -- dtype size is 8 for complex32 and 16 for complex64
    format.println("Size of dtype is {}", dtype_size)

    -- Call fftw_c.fftw_plan_dft: fftw_plan_dft_1d(int n, fftw_complex *in,
    -- fftw_complex *out,int sign, unsigned flags). n is the size of transform,
    -- in and out are pointers to the input and output arrays. Sign is the sign
    -- of the exponent in the transform, can either be FFTW_FORWARD (1) or
    -- FFTW_BACKWARD (-1). Flags: FFTW_ESTIMATE, on the contrary, does not run
    -- any computation
    format.println("Calling fftw_plan_dft to store fftw_plan in p.p...")

    -- R2C: Float to Complex32
    if dtype_size == 8 and real_flag then
      format.println("R2C: Float to Complex32: Not Supported")
      -- AK Note: I don't actually understand this that well, how the size of each dimension is stored in the 'n' array
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)] --n is a array of size dim storing the size of each dimension
      format.println("calling fftwf_plan_dft_r2c")
      -- p.float_p = fftw_c.fftwf_plan_dft_r2c(dim, &n[0], [&float](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_ESTIMATE)

    elseif dtype_size == 8 then
      format.println("R2C: Complex32 to Complex32: Not Supported")
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      format.println("calling fftwf_plan_dft")
      -- p.float_p = fftw_c.fftwf_plan_dft(dim, &n[0], [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)

    elseif dtype_size == 16 and real_flag then
      format.println("R2C: Double to Complex64: Calling fftw_plan_dft_r2c")
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      p.p = fftw_c.fftw_plan_dft_r2c(dim, &n[0], [&double](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_ESTIMATE)

    elseif dtype_size == 16 then
      format.println("R2C: Complex64 to Complex64: Calling fftw_plan_dft_r2c")
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      format.println("n[0] is {}, dim is {}", n[0], dim)
      p.p = fftw_c.fftw_plan_dft(dim, &n[0], [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    end

    p.address_space = address_space

    rescape
      if gpu_available then
        remit requote
          format.println("Num of local GPUs: {}", iface.get_num_local_gpus())
          if iface.get_num_local_gpus() > 0 then
            format.println("GPUs identified: calling make_plan_gpu...")
            make_plan_gpu(input, output, plan, p.address_space)
          end
        end
      end
    end
  end

  -- Task: make_plan_batch in GPU version. Calls cufftPlanMany and stores plan in cufft_p
  local make_plan_gpu_batch
  rescape
    if gpu_available then
      remit rquote
        __demand(__cuda, __leaf)
        task make_plan_gpu_batch(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan), address_space : c.legion_address_space_t)
        where reads writes(input, output, plan) do
          format.println("In iface.make_plan_gpu_batch...")

          var p = iface.get_plan(plan, true)

          -- Verify we are in GPU mode by checking TOC_PROC
          -- Takes a c.legion_runtime_t and returns c.legion_runtime_get_executing_processor(runtime, ctx)
          var proc = get_executing_processor(__runtime())
          format.println("Make_Plan_GPU: TOC PROC IS {}",c.TOC_PROC)
          format.println("Make_Plan_GPU: Processor kind is {}", c.legion_processor_kind(proc))

          if c.legion_processor_kind(proc) == c.TOC_PROC then
            format.println("Processor is TOC, so running GPU functions")
            var i = c.legion_processor_address_space(proc)
            regentlib.assert(address_space == i, "make_plan_gpu_batch must be executed on a processor in the same address space")

            var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
            var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
            var lo = input.ispace.bounds.lo:to_point()
            var hi = input.ispace.bounds.hi:to_point()
            var n : int[dim]
            ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

            -- Define 'n' array: n is an array of size rank, describing the size of
            -- each dimension. For batched transforms, we want to exclude the last
            -- dimension as that is the number of batches.
            --
            -- AK Note: There must be a better way to copy/slice arrays in regent
            -- instead of naively using this for loop.
            var n_batch : int[dim-1]
            for i = 0, dim do
              n_batch[i] = n[i]
            end

            -- Set idist: idist is the distance between the first element of two
            -- consecutive batches. In a transform where each batch is a 256x256
            -- complex64 transform, offset_1 will be 16, offset_2 will be 16*256,
            -- and offet_3 willbe 16*256*256. idist should be 256*256 in this case,
            -- so we want offset_3/offset_1.
            var offset_in = get_offset_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
            var offset_1 = offset_in[0].offset
            var offset_2 = offset_in[1].offset
            var offset_3 = offset_in[2].offset
            var i_dist = offset_3/offset_1

            format.println("n[0] = {}, n[1] = {}, n[2] = {}, n_batch[0] = {}, n_batch[1] = {}, i_dist = {}", n[0], n[1], n[2], n_batch[0], n_batch[1], i_dist)

            -- Call cufftPlanMany: cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch) --rank = dimensionality of transform (1,2,3)

            -- Taking a transform where {256x256x7} is passed (i.e. 7 batches of 256x256), the parameters should be as follows:
            -- rank[In] – How many dimensions a single transform has: 2
            -- n[In] - Array of size rank, describing the size of each dimension: {256, 256}
            -- inembed[In] – Array of size rank. Here we pass the real dimensions of the input array. It should contain the dimension size + padding: {256, 256} again, because input is not padded.
            -- istride[In] – Stride between two consecutive elements in lowest dimension: 1
            -- idist[In] – Distance between two input batches: 256 * 256
            -- onembed[In] – Array of size rank. Here we pass the real dimensions of the output array. It should contain the dimension size + padding: {256, 256} again, because input is not padded.
            -- ostride[In] – Stride between two consecutive elements in lowest dimension: 1
            -- odist[In] – Distance between two output batches: 256 * 256
            -- type[In] – The transform data type: Z2Z or D2Z.
            -- batch[In] – How many batches should be computed: 7

            format.println("Calling cufftPlanMany for batched transform...")
            var ok = 0

            -- R2C: Float to Complex32
            if dtype_size == 8 and real_flag then
              format.println("Calling cufftPlanMany with CUFFT_R2C: Float to Complex32...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], 1, i_dist, &n_batch[0], 1, i_dist, cufft_c.CUFFT_R2C, n[dim-1])

            -- C2C: Complex32 to Complex32
            elseif dtype_size == 8 then
              format.println("Calling cufftPlanMany with CUFFT_C2C: Complex32 to Complex32 ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], 1, i_dist, &n_batch[0], 1, i_dist, cufft_c.CUFFT_C2C, n[dim-1])

            -- R2C: Double to Complex64
            elseif real_flag and dtype_size == 16 then
              format.println("Calling cufftPlanMany with CUFFT_D2Z: Double to Complex64 ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], 1, i_dist, &n_batch[0], 1, i_dist, cufft_c.CUFFT_D2Z, n[dim-1])

            -- C2C: Complex64 to Complex64
            elseif dtype_size == 16 then
              format.println("Calling cufftPlanMany with CUFFT_Z2Z: Complex64 to Complex64 ...")
              ok = cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], 1, i_dist, &n_batch[0], 1, i_dist, cufft_c.CUFFT_Z2Z, n[dim-1])
            end

            -- Check return value of cufftPlanMany, throw error if necessary
            if ok == cufft_c.CUFFT_INVALID_VALUE then
              format.println("Invalid value in cufftPlanMany")
            end
            regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftPlanMany failed")
            format.println("cufftPlanMany Successful")

          -- GPU not identified: return error
          else
            format.println("GPU processor not identified: TOC_PROC not equal to processor kind")
            regentlib.assert(false, "make_plan_gpu must be executed on a GPU processor")
          end
        end
      end
    end
  end

  --- Make plan (batched version).
  -- @param input Input region.
  -- @param output Output region.
  -- @param plan Plan region.
  -- @note Calls make_plan_gpu_batch if necessary.
  __demand(__inline)
  task iface.make_plan_batch(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    format.println("In iface.make_plan_batch...")

    -- Get pointer to plan
    format.println("Calling get_plan...")
    var p = iface.get_plan(plan, false)
    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime()))
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")

    var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])
    var offset_in = get_offset_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])

    -- Set idist: idist is the distance between the first element of two
    -- consecutive batches. In a transform where each batch is a 256x256
    -- complex64 transform, offset_1 will be 16, offset_2 will be 16*256, and
    -- offet_3 willbe 16*256*256. idist should be 256*256 in this case, so we
    -- want offset_3/offset_1.
    var offset_1 = offset_in[0].offset
    var offset_2 = offset_in[1].offset
    var offset_3 = offset_in[2].offset
    var i_dist = offset_3/offset_1

    format.println("Offset 1 = {}, Offset 2 = {}, Offset 3 = {}", offset_1, offset_2, offset_3)

    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()

    -- dtype size is 8 for complex32 and 16 for complex64
    format.println("Size of dtype is {}", dtype_size)

    -- If GPUs, call make_plan_gpu_batch

    rescape
      if gpu_available then
        remit rquote
          format.println("Num_local_gpus is {}", iface.get_num_local_gpus())
          if iface.get_num_local_gpus() > 0 then
            format.println("GPUs identified: calling make_plan_gpu_batch...")
            make_plan_gpu_batch(input, output, plan, p.address_space)
          end
        end
      end
    end

    -- Call fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags)
    -- Taking a transform where {256x256x7} is passed (i.e. 7 batches of 256x256), the parameters should be as follows:
    -- rank[In] – How many dimensions a single transform has: 2
    -- n[In] - Array of size rank, describing the size of each dimension: {256, 256}
    -- inembed[In] – Array of size rank. Here we pass the real dimensions of the input array. It should contain the dimension size + padding: {256, 256} again, because input is not padded.
    -- istride[In] – Stride between two consecutive elements in lowest dimension: 1
    -- idist[In] – Distance between two input batches: 256 * 256
    -- onembed[In] – Array of size rank. Here we pass the real dimensions of the output array. It should contain the dimension size + padding: {256, 256} again, because input is not padded.
    -- ostride[In] – Stride between two consecutive elements in lowest dimension: 1
    -- odist[In] – Distance between two output batches: 256 * 256
    -- type[In] – The transform data type: Z2Z or D2Z.
    -- batch[In] – How many batches should be computed: 7

    -- R2C: Float to Complex32
    if dtype_size == 8 and real_flag then
      format.println("R2C: Float to Complex32: Not Supported for CPU mode")
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      format.println("calling fftwf_plan_dft_r2c")
      --p.float_p = fftw_c.fftwf_plan_dft_r2c(dim, &n[0], [&float](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_ESTIMATE) --Commented out because not supported: FFTW does not support float and double on single install

    -- C2C: Complex32 to Complex32
    elseif dtype_size == 8 then
      format.println("R2C: Complex32 to Complex32: Not Supported for CPU mode")
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]
      format.println("calling fftwf_plan_dft")
      -- p.float_p = fftw_c.fftwf_plan_dft(dim, &n[0], [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE) --Commented out because not supported: FFTW does not support float and double on single install

    -- R2C: Double to Complex64
    elseif dtype_size == 16 and real_flag then
      format.println("R2C: Double to Complex64: Calling fftw_plan_dft_r2c")
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

      -- Define 'n' array: n is an array of size rank, describing the size of
      -- each dimension. For batched transforms, we want to exclude the last
      -- dimension as that is the number of batches
      --
      -- AK Note: There must be a better way to copy/slice arrays in regent
      -- instead of naively using this for loop
      var n_batch : int[dim-1]
      for i = 0, dim do
        n_batch[i] = n[i]
      end

      format.println("fftw_plan_many_dft_r2c: n[0] = {}, n[1] = {}, n[2] = {}, n_batch[0] = {}, n_batch[1] = {}, i_dist = {}", n[0], n[1], n[2], n_batch[0], n_batch[1], i_dist)

      -- AK Note: This is segfaulting now. Complex64 to Complex64 works somehow,
      -- even though the call is the same
      p.p = fftw_c.fftw_plan_many_dft_r2c(dim-1, &n_batch[0], n[dim-1], [&double](input_base), &n_batch[0], 1, i_dist, [&fftw_c.fftw_complex](output_base), &n_batch[0], 1, i_dist, fftw_c.FFTW_ESTIMATE)

    -- C2C: Complex64 to Complex64
    elseif dtype_size == 16 then
      var n : int[dim]
      ;[data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

      -- Define 'n' array: n is an array of size rank, describing the size of
      -- each dimension. For batched transforms, we want to exclude the last
      -- dimension as that is the number of batches
      --
      -- AK Note: There must be a better way to copy/slice arrays in regent
      -- instead of naively using this for loop
      var n_batch : int[dim-1]
      for i = 0, dim do
        n_batch[i] = n[i]
      end

      format.println("n[0] = {}, n[1] = {}, n[2] = {}, n_batch[0] = {}, n_batch[1] = {}, i_dist = {}", n[0], n[1], n[2], n_batch[0], n_batch[1], i_dist)
      p.p = fftw_c.fftw_plan_many_dft(dim-1, &n_batch[0], n[dim-1], [&fftw_c.fftw_complex](input_base), &n_batch[0], 1, i_dist, [&fftw_c.fftw_complex](output_base), &n_batch[0], 1, i_dist,  fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    end
    p.address_space = address_space
  end

  --- Make plan task.
  -- @param input Input region.
  -- @param output Output region.
  -- @param plan Plan region.
  task iface.make_plan_task(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    iface.make_plan(input, output, plan)
  end

  --- Make plan (distributed version).
  -- @param input Input region.
  -- @param input_part Input partition.
  -- @param output Output region.
  -- @param output_part Output partition.
  -- @param plan Plan region.
  -- @param plan_part Plan partition.
  __demand(__inline)
  task iface.make_plan_distrib(input : region(ispace(itype), dtype_in), input_part : partition(disjoint, input, ispace(int1d)), output : region(ispace(itype), dtype_out), output_part : partition(disjoint, output, ispace(int1d)), plan : region(ispace(int1d), iface.plan), plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(input, output, plan) do

    -- Get number of nodes and check consistency of nodes/colors
    var n = iface.get_num_nodes()
    regentlib.assert(input_part.colors.bounds.hi - input_part.colors.bounds.lo + 1 == int1d(n), "input_part colors size must be equal to the number of nodes")
    regentlib.assert(input_part.colors.bounds == output_part.colors.bounds, "input_part and output_part colors must be equal")
    regentlib.assert(input_part.colors.bounds == plan_part.colors.bounds, "input_part and plan_part colors must be equal")

    var p : iface.plan
    -- T(x) is a cast from type T to a value x
    p.p = [fftw_c.fftw_plan](0)

    
    rescape
      if gpu_available then
        remit rquote
          p.cufft_p = 0
        end
      end
    end

    fill(plan, p)

    __demand(__index_launch)
    for i in plan_part.colors do
      iface.make_plan_task(input_part[i], output_part[i], plan_part[i])
    end
  end

  -- EXECUTE PLAN FUNCTIONS

  --- Execute plan.
  -- @param input Input to execute plan on.
  -- @param output Output of the executed plan.
  -- @param plan Plan object used to execute plan.
  -- @note For GPU, it calls cufftExecZ2Z.
  -- @note For CPU, it calls fftw_execute_dft.
   __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), writes(output) do
    format.println("In iface.execute_plan...")

    var p = iface.get_plan(plan, true)
    var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0])
    var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0])

    -- Check if we are in GPU or CPU mode
    var proc = get_executing_processor(__runtime())
    format.println("execute_plan: TOC PROC IS {}", c.TOC_PROC) -- TOC = Throughput Oriented Core: Means we have a GPU
    format.println("execute_plan: Processor kind is {}", c.legion_processor_kind(proc))
    format.println("size of dtype is {}", dtype_size) -- dtype size is 8 for complex32 and 16 for complex64

    -- If in GPU mode, use cufftExec
    if c.legion_processor_kind(proc) == c.TOC_PROC then
      c.printf("execute plan via cuFFT\n")
      var ok = 0

      -- dtype size is 8 for complex32 and 16 for complex64
      format.println("size of dtype is {}", dtype_size)

      -- R2C float to complex32
      if dtype_size == 8 and real_flag then
        format.println("Calling cufftExecR2C (Float to Complex32) ...")
        -- ok = cufft_c.cufftExecR2C(p.cufft_p, [&cufft_c.cufftReal](input_base), [&cufft_c.cufftComplex](output_base))
      -- C2C complex32 to complex32
      elseif dtype_size == 8 then
        format.println("Calling cufftExecC2C (Complex32 to Complex32)...")
        ok = cufft_c.cufftExecC2C(p.cufft_p, [&cufft_c.cufftComplex](input_base), [&cufft_c.cufftComplex](output_base), cufft_c.CUFFT_FORWARD)
      -- R2C double to complex64
      elseif dtype_size == 16 and real_flag then
        format.println("Calling cufftExecD2Z (Double to Complex64)...")
        ok = cufft_c.cufftExecD2Z(p.cufft_p, [&cufft_c.cufftDoubleReal](input_base), [&cufft_c.cufftDoubleComplex](output_base))
      -- C2C complex64 to complex64
      elseif dtype_size == 16 then
        format.println("Calling cufftExecZ2Z (Complex64 to Complex64)...")
        ok = cufft_c.cufftExecZ2Z(p.cufft_p, [&cufft_c.cufftDoubleComplex](input_base), [&cufft_c.cufftDoubleComplex](output_base), cufft_c.CUFFT_FORWARD)
      end

      -- Check return values of Exec
      if ok == cufft_c.CUFFT_INVALID_VALUE then
        format.println("Invalid value in cufftExecZ2Z")
      elseif ok == cufft_c.CUFFT_INVALID_PLAN then
        format.println("Invalid plan passed to cufftExecZ2Z")
      end

      -- format.println("cufftExecZ2Z returned {}", ok)
      regentlib.assert(ok == cufft_c.CUFFT_SUCCESS, "cufftExecZ2Z failed")
      format.println("cufftExecZ2Z successful")

    -- Otherwise, we are in CPU mode: use FFTW if no GPU
    else
      c.printf("execute plan via FFTW\n")
      --R2C float to complex32: Not supported
      if dtype_size == 8 and real_flag then
        format.println("Executing FFTW R2C Float to Complex32: Not Supported")
        --fftw_c.fftwf_execute_dft_r2c(p.float_p, [&float](input_base), [&fftw_c.fftwf_complex](output_base))
      -- C2C complex32 to complex32: Not supported
      elseif dtype_size == 8 then
        format.println("Executing FFTW C2C Complex32 to Complex32: Not Supported")
        -- fftw_c.fftwf_execute_dft(p.float_p, [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base))
      -- R2C double to complex64
      elseif dtype_size == 16 and real_flag then
        format.println("Executing FFTW R2C double to complex64")
        fftw_c.fftw_execute_dft_r2c(p.p, [&double](input_base), [&fftw_c.fftw_complex](output_base))
      -- C2C complex64 to complex64
      elseif dtype_size == 16 then
        format.println("Executing FFTW C2C complex64 to complex64")
        fftw_c.fftw_execute_dft(p.p, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base))
      end
    end
  end

  --- Execute plan task.
  -- @param input Input to execute plan on.
  -- @param output Output of the executed plan.
  -- @param plan Plan object used to execute plan.
  __demand(__cuda, __leaf)
  task iface.execute_plan_task(input : region(ispace(itype), dtype_in), output : region(ispace(itype), dtype_out), plan : region(ispace(int1d), iface.plan))
  where reads(input, plan), reads writes(output) do
    iface.execute_plan(input, output, plan)
  end

  -- DESTROY PLAN FUNCTIONS

  --- Destroy plan.
  -- @param plan Plan to be destroyed.
  -- @note To execute it in a separate task, it must be wrapped into a task.
  -- @note That is what `destroy_plan_task` does.
  __demand(__inline)
  task iface.destroy_plan(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
   format.println("In iface.destroy_plan...")

    var p = iface.get_plan(plan, true)
    var proc = get_executing_processor(__runtime())

    -- If using GPUs, call cufftDestroy
    if c.legion_processor_kind(proc) == c.TOC_PROC then
      c.printf("Destroy plan via cuFFT\n")
      -- Function: cufftResult cufftDestroy(cufftHandle plan)
      cufft_c.cufftDestroy(p.cufft_p)
    else
      -- Else, if on CPUs, call fftw_destroy
      c.printf("Destroy plan via FFTW\n")
      fftw_c.fftw_destroy_plan(p.p)
      -- Commented out float version of FFTW: not supported
      -- fftw_c.fftwf_destroy_plan(p.float_p)
    end
  end

  --- Destroy plan task.
  -- @param plan Plan to be destroyed.
  task iface.destroy_plan_task(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    iface.destroy_plan(plan)
  end

  --- Destroy plan (distributed version).
  -- @param plan Plan to be destroyed.
  -- @param plan_part Plan partition to be destroyed in `plan`.
  __demand(__inline)
  task iface.destroy_plan_distrib(plan : region(ispace(int1d), iface.plan), plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(plan) do
    format.println("In iface.destroy_plan_distrib...")
    for i in plan_part.colors do
      iface.destroy_plan_task(plan_part[i])
    end
  end

  return iface
end

return fft
