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

--[[--
Regent FFT library.
]]

import "regent"

local format = require("std/format")
local data = require("common/data")
local gpuhelper = require("regent/gpu/helper")
local gpu_available = gpuhelper.check_gpu_available()

-- Import C and FFTW APIs
local c = regentlib.c
local fftw_c = terralib.includec("fftw3.h")
regentlib.linklibrary("libfftw3.so")
regentlib.linklibrary("libfftw3f.so")

-- Import cuFFT API
local cufft_c
local cufft_assert
if gpu_available then
  cufft_c = terralib.includec("cufftXt.h")
  regentlib.linklibrary("libcufft.so")

  terra cufft_assert(result : cufft_c.cufftResult)
    var status = "UNKNOWN"
    if result == cufft_c.CUFFT_SUCCESS then
      status = "CUFFT_SUCCESS"
    elseif result == cufft_c.CUFFT_INVALID_PLAN then
      status = "CUFFT_INVALID_PLAN"
    elseif result == cufft_c.CUFFT_ALLOC_FAILED then
      status = "CUFFT_ALLOC_FAILED"
    elseif result == cufft_c.CUFFT_INVALID_TYPE then
      status = "CUFFT_INVALID_TYPE"
    elseif result == cufft_c.CUFFT_INVALID_VALUE then
      status = "CUFFT_INVALID_VALUE"
    elseif result == cufft_c.CUFFT_INTERNAL_ERROR then
      status = "CUFFT_INTERNAL_ERROR"
    elseif result == cufft_c.CUFFT_EXEC_FAILED then
      status = "CUFFT_EXEC_FAILED"
    elseif result == cufft_c.CUFFT_SETUP_FAILED then
      status = "CUFFT_SETUP_FAILED"
    elseif result == cufft_c.CUFFT_INVALID_SIZE then
      status = "CUFFT_INVALID_SIZE"
    elseif result == cufft_c.CUFFT_UNALIGNED_DATA then
      status = "CUFFT_UNALIGNED_DATA"
    elseif result == cufft_c.CUFFT_INCOMPLETE_PARAMETER_LIST then
      status = "CUFFT_INCOMPLETE_PARAMETER_LIST"
    elseif result == cufft_c.CUFFT_INVALID_DEVICE then
      status = "CUFFT_INVALID_DEVICE"
    elseif result == cufft_c.CUFFT_PARSE_ERROR then
      status = "CUFFT_PARSE_ERROR"
    elseif result == cufft_c.CUFFT_NO_WORKSPACE then
      status = "CUFFT_NO_WORKSPACE"
    elseif result == cufft_c.CUFFT_NOT_IMPLEMENTED then
      status = "CUFFT_NOT_IMPLEMENTED"
    elseif result == cufft_c.CUFFT_LICENSE_ERROR then
      status = "CUFFT_LICENSE_ERROR"
    elseif result == cufft_c.CUFFT_NOT_SUPPORTED then
      status = "CUFFT_NOT_SUPPORTED"
    end
    regentlib.assert(result == cufft_c.CUFFT_SUCCESS, status)
  end
end

-- Hack: get defines from fftw3.h
fftw_c.FFTW_FORWARD = -1
fftw_c.FFTW_BACKWARD = 1

fftw_c.FFTW_MEASURE = 0
fftw_c.FFTW_DESTROY_INPUT = (2 ^ 0)
fftw_c.FFTW_UNALIGNED = (2 ^ 1)
fftw_c.FFTW_CONSERVE_MEMORY = (2 ^ 2)
fftw_c.FFTW_EXHAUSTIVE = (2 ^ 3) -- NO_EXHAUSTIVE is default
fftw_c.FFTW_PRESERVE_INPUT = (2 ^ 4) -- cancels FFTW_DESTROY_INPUT
fftw_c.FFTW_PATIENT = (2 ^ 5) -- IMPATIENT is default
fftw_c.FFTW_ESTIMATE = (2 ^ 6)
fftw_c.FFTW_WISDOM_ONLY = (2 ^ 21)

local fft = {}

--- Create an FFT interface.
-- @param itype_input Index type of transform (int1d/int2d/int3d).
-- @param dtype_in Input data type of transform (float/double/complex32/complex64).
-- @param dtype_out Output data type of transform (complex32/complex64).
-- @param batch_flag Flag for batch transform
function fft.generate_fft_interface(itype_input, dtype_in, dtype_out, batch_flag)
  assert(regentlib.is_index_type(itype_input), "requires an index type as the first argument")
  local dim = itype_input.dim
  local dtype_out_size = terralib.sizeof(dtype_out)
  assert(dim >= 1 and dim <= 3, "currently only 1 <= dim <= 3 is supported")
  assert(dtype_in == float or dtype_in == double or dtype_in == complex32 or dtype_in == complex64, "input type must be float/double/complex32/complex64")
  assert(dtype_out == complex32 or dtype_out == complex64, "output type must be complex32/complex64")

  local itype = itype_input
  if batch_flag then
    dim = dim + 1
    if itype == int1d then
      itype = int2d
    elseif itype == int2d then
      itype = int3d
    elseif itype == int3d then
      itype = int4d
    end
  end

   -- Single-precision transforms
  local float_to_complex32_transform = (dtype_in == float and dtype_out == complex32)
  local complex32_to_complex32_transform = (dtype_in == complex32 and dtype_out == complex32)

   -- Double-precision transforms
  local double_to_complex64_transform = (dtype_in == double and dtype_out == complex64)
  local complex64_to_complex64_transform = (dtype_in == complex64 and dtype_out == complex64)

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
  iface.plan.__no_field_slicing = true -- Don't field slice this struct

  local function make_get_base(d, t)
    local rect_t = c["legion_rect_" .. d .. "d_t"]
    local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. d .. "d"]
    local raw_rect_ptr = c["legion_accessor_array_" .. d .. "d_raw_rect_ptr"]
    local destroy_accessor = c["legion_accessor_array_" .. d .. "d_destroy"]

    local struct base_pointer_t {
      base : &t,
      offset : c.legion_byte_offset_t[d],
      dtype_size : int
    }

    local terra get_base(rect : rect_t,
                         physical : c.legion_physical_region_t,
                         field : c.legion_field_id_t)
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
      regentlib.assert(offsets[0].offset == terralib.sizeof(t), "stride does not match expected value")
      destroy_accessor(accessor)

      var bp : base_pointer_t
      bp.base = base_pointer
      bp.offset = offsets
      bp.dtype_size = terralib.sizeof(t)
      return bp
    end

    return rect_t, get_base
  end

  -- Define get_base functions for input, output, and plan regions
  local rect_plan_t, get_base_plan = make_get_base(1, iface.plan)
  local rect_in_t, get_base_in = make_get_base(dim, dtype_in)
  local rect_out_t, get_base_out = make_get_base(dim, dtype_out)

  -- Used to branch on CPU vs GPU execution inside tasks
  -- Hack: Need to retrieve context without __context() to circumvent leaf checker here.
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

    -- FIXME (Elliott): I thought Regent was supposed to copy on
    -- assignment, but that seems not to happen here, so this would
    -- result in a double destroy if we free here.

    -- c.legion_future_destroy(f)
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
    -- Hack: Need to use raw access to circument CUDA checker here.
    var pr = __physical(plan)[0]
    regentlib.assert(c.legion_physical_region_get_memory_count(pr) == 1, "plan instance has more than one memory?")
    var mem_kind = c.legion_memory_kind(c.legion_physical_region_get_memory(pr, 0))
    regentlib.assert(
      mem_kind == c.SYSTEM_MEM or mem_kind == c.REGDMA_MEM or mem_kind == c.Z_COPY_MEM,
      "plan instance must be stored in sysmem, regmem, or zero copy mem")
    var plan_base = get_base_plan(rect_plan_t(plan.ispace.bounds), __physical(plan)[0], __fields(plan)[0]).base
    var i = c.legion_processor_address_space(get_executing_processor(__runtime()))
    var p : &iface.plan
    var bounds = plan.ispace.bounds
    if bounds.hi - bounds.lo + 1 > int1d(1) then
      p = plan_base + i
    else
      p = plan_base
    end
    regentlib.assert(not check or p.address_space == i, "plans can only be used on the node where they are originally created")
    return p
  end

  -- MAKE PLAN FUNCTIONS

  -- Creates the FFT plan - GPU version. Called by make_plan if necessary.
  local make_plan_gpu
  if gpu_available then
    __demand(__cuda, __leaf)
    task make_plan_gpu(input : region(ispace(itype), dtype_in),
                       output : region(ispace(itype), dtype_out),
                       plan : region(ispace(int1d), iface.plan),
                       address_space : c.legion_address_space_t)
    where reads writes(input, output, plan) do

      var p = iface.get_plan(plan, true)
      var proc = get_executing_processor(__runtime())

      regentlib.assert(c.legion_processor_kind(proc) == c.TOC_PROC, "make_plan_gpu must be executed on a GPU processor")
      regentlib.assert(c.legion_processor_address_space(proc) == address_space, "make_plan_gpu must be executed on a processor in the same address space")

      -- Get input and output bases
      var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).base
      var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0]).base
      var lo = input.ispace.bounds.lo:to_point()
      var hi = input.ispace.bounds.hi:to_point()
      var n : int[dim] -- n is an array of size dim with the size of each dimension in the entries
      [data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

      -- Create plans
      if float_to_complex32_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_R2C, 1))
      elseif complex32_to_complex32_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_C2C, 1))
      elseif double_to_complex64_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_D2Z, 1))
      elseif complex64_to_complex64_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim, &n[0], [&int](0), 0, 0, [&int](0), 0, 0, cufft_c.CUFFT_Z2Z, 1))
      end
    end
  end

  --- Creates the FFT plan. This __inline version will execute in the caller's context. Supports both CPU and GPU.
  -- @param input Input region.
  -- @param output Output region.
  -- @param plan Plan region.
   __demand(__inline)
  task iface.make_plan(input : region(ispace(itype), dtype_in),
                       output : region(ispace(itype), dtype_out),
                       plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do

    var p = iface.get_plan(plan, false)
    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime()))
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")

    var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).base
    var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0]).base
    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()

    var n : int[dim]
    [data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

    if float_to_complex32_transform then
      p.float_p = fftw_c.fftwf_plan_dft_r2c(dim, &n[0], [&float](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_ESTIMATE)
    elseif complex32_to_complex32_transform then
      p.float_p = fftw_c.fftwf_plan_dft(dim, &n[0], [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    elseif double_to_complex64_transform then
      p.p = fftw_c.fftw_plan_dft_r2c(dim, &n[0], [&double](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_ESTIMATE)
    elseif complex64_to_complex64_transform then
      p.p = fftw_c.fftw_plan_dft(dim, &n[0], [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    end

    p.address_space = address_space

    rescape
      if gpu_available then
        remit rquote
          if iface.get_num_local_gpus() > 0 then
            make_plan_gpu(input, output, plan, p.address_space)
          end
        end
      end
    end
  end

  -- Creates the FFT plan - GPU + batch version. Called by make_plan_batch if necessary.
  local make_plan_gpu_batch
  if gpu_available then
    __demand(__cuda, __leaf)
    task make_plan_gpu_batch(input : region(ispace(itype), dtype_in),
                             output : region(ispace(itype), dtype_out),
                             plan : region(ispace(int1d), iface.plan),
                             address_space : c.legion_address_space_t)
    where reads writes(input, output, plan) do

      var p = iface.get_plan(plan, true)
      var proc = get_executing_processor(__runtime())
      regentlib.assert(c.legion_processor_kind(proc) == c.TOC_PROC, "make_plan_gpu_batch must be executed on a GPU processor")
      regentlib.assert(c.legion_processor_address_space(proc) == address_space, "make_plan_gpu_batch must be executed on a processor in the same address space")

      var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).base
      var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0]).base
      var lo = input.ispace.bounds.lo:to_point()
      var hi = input.ispace.bounds.hi:to_point()

      var n : int[dim]
      [data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

      -- For batched transforms, we want to exclude the last dimension as that
      -- is the number of batches.
      var n_batch : int[dim-1]
      for i = 0, dim do
        n_batch[i] = n[i]
      end

      var offset_in = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).offset
      var dtype_size_in = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).dtype_size
      var offset_1 = offset_in[0].offset
      var offset_2 = offset_in[1].offset
      var offset_3 = offset_in[2].offset
      var i_dist : int

      if dim == 2 then
        i_dist = offset_2 / offset_1
      elseif dim == 3 then
        i_dist = offset_3 / offset_1
      elseif dim == 4 then
        i_dist = offset_in[3].offset / offset_1
      end

      var istride = offset_in[0].offset / dtype_size_in

      if float_to_complex32_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], istride, i_dist, &n_batch[0], istride, i_dist, cufft_c.CUFFT_R2C, n[dim-1]))
      elseif complex32_to_complex32_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], istride, i_dist, &n_batch[0], istride, i_dist, cufft_c.CUFFT_C2C, n[dim-1]))
      elseif double_to_complex64_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], istride, i_dist, &n_batch[0], istride, i_dist, cufft_c.CUFFT_D2Z, n[dim-1]))
      elseif complex64_to_complex64_transform then
        cufft_assert(cufft_c.cufftPlanMany(&p.cufft_p, dim-1, &n_batch[0], &n_batch[0], istride, i_dist, &n_batch[0], istride, i_dist, cufft_c.CUFFT_Z2Z, n[dim-1]))
      end
    end
  end

  --- Make plan (batched version).
  -- @param input Input region.
  -- @param output Output region.
  -- @param plan Plan region.
  -- @note Calls make_plan_gpu_batch if necessary.
  __demand(__inline)
  task iface.make_plan_batch(input : region(ispace(itype), dtype_in),
                             output : region(ispace(itype), dtype_out),
                             plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do

    var p = iface.get_plan(plan, false)

    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime()))
    regentlib.assert(input.ispace.bounds == output.ispace.bounds, "input and output regions must be identical in size")

    var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).base
    var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0]).base
    var offset_in = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).offset
    var dtype_in_size = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).dtype_size

    -- Set idist: idist is the distance between the first element of two
    -- consecutive batches. In a transform where each batch is a 256x256
    -- complex64 transform, offset_1 will be 16, offset_2 will be 16*256, and
    -- offet_3 willbe 16*256*256. idist should be 256*256 in this case, so we
    -- want offset_3/offset_1.
    var offset_1 = offset_in[0].offset
    var offset_2 = offset_in[1].offset
    var offset_3 = offset_in[2].offset

    var i_dist : int
    if dim == 2 then
      i_dist = offset_2/offset_1
    elseif dim == 3 then
      i_dist = offset_3/offset_1
    elseif dim == 4 then
      i_dist = offset_in[3].offset/offset_1
    end

    var istride = offset_1 / dtype_in_size

    var lo = input.ispace.bounds.lo:to_point()
    var hi = input.ispace.bounds.hi:to_point()

    p.address_space = address_space

    var n : int[dim]
    [data.range(dim):map(function(i) return rquote n[i] = hi.x[i] - lo.x[i] + 1 end end)]

    -- For batched transforms, we want to exclude the last dimension as that is
    -- the number of batches.
    var n_batch : int[dim-1]
    for i = 0, dim do
      n_batch[i] = n[i]
    end

    if float_to_complex32_transform then
      p.float_p = fftw_c.fftwf_plan_dft_r2c(dim, &n[0], [&float](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_ESTIMATE)
    elseif complex32_to_complex32_transform then
      p.float_p = fftw_c.fftwf_plan_dft(dim, &n[0], [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base), fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    elseif double_to_complex64_transform then
      p.p = fftw_c.fftw_plan_many_dft_r2c(dim-1, &n_batch[0], n[dim-1], [&double](input_base), &n_batch[0], istride, i_dist, [&fftw_c.fftw_complex](output_base), &n_batch[0], istride, i_dist, fftw_c.FFTW_ESTIMATE)
    elseif complex64_to_complex64_transform then
      p.p = fftw_c.fftw_plan_many_dft(dim-1, &n_batch[0], n[dim-1], [&fftw_c.fftw_complex](input_base), &n_batch[0], istride, i_dist, [&fftw_c.fftw_complex](output_base), &n_batch[0], istride, i_dist,  fftw_c.FFTW_FORWARD, fftw_c.FFTW_ESTIMATE)
    end

    rescape
      if gpu_available then
        remit rquote
          if iface.get_num_local_gpus() > 0 then
            make_plan_gpu_batch(input, output, plan, p.address_space)
          end
        end
      end
    end
  end

  --- Creates the plan. This version will launch a task.
  -- @param input Input region.
  -- @param output Output region.
  -- @param plan Plan region.
  task iface.make_plan_task(input : region(ispace(itype), dtype_in),
                            output : region(ispace(itype), dtype_out),
                            plan : region(ispace(int1d), iface.plan))
  where reads writes(input, output, plan) do
    iface.make_plan(input, output, plan)
  end

  --- Create the plan. This version launches `make_plan_task` for each subregion in the provided partitions.
  -- @param input Input region.
  -- @param input_part Input partition.
  -- @param output Output region.
  -- @param output_part Output partition.
  -- @param plan Plan region.
  -- @param plan_part Plan partition.
  __demand(__inline)
  task iface.make_plan_distrib(input : region(ispace(itype), dtype_in),
                               input_part : partition(disjoint, input, ispace(int1d)),
                               output : region(ispace(itype), dtype_out),
                               output_part : partition(disjoint, output, ispace(int1d)),
                               plan : region(ispace(int1d), iface.plan),
                               plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(input, output, plan) do

    -- Get number of nodes and check consistency of nodes/colors
    var n = iface.get_num_nodes()
    regentlib.assert(input_part.colors.bounds.hi - input_part.colors.bounds.lo + 1 == int1d(n), "input_part colors size must be equal to the number of nodes")
    regentlib.assert(input_part.colors.bounds == output_part.colors.bounds, "input_part and output_part colors must be equal")
    regentlib.assert(input_part.colors.bounds == plan_part.colors.bounds, "input_part and plan_part colors must be equal")

    var p : iface.plan
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

  -- Executes the FFT plan - GPU version. Called by execute_plan if necessary.
  local execute_plan_gpu
  if gpu_available then
    __demand(__cuda, __leaf)
    task execute_plan_gpu(input : region(ispace(itype), dtype_in),
                          output : region(ispace(itype), dtype_out),
                          plan : region(ispace(int1d), iface.plan),
                          address_space : c.legion_address_space_t)
    where reads writes (input, output, plan) do

      var p = iface.get_plan(plan, true)
      var proc = get_executing_processor(__runtime())
      regentlib.assert(c.legion_processor_kind(proc) == c.TOC_PROC, "execute_plan_gpu must be executed on a GPU processor")
      regentlib.assert(c.legion_processor_address_space(proc) == address_space, "execute_plan_gpu must be executed on a processor in the same address space")

      var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).base
      var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0]).base

      if float_to_complex32_transform then
        cufft_assert(cufft_c.cufftExecR2C(p.cufft_p, [&cufft_c.cufftReal](input_base), [&cufft_c.cufftComplex](output_base)))
      elseif complex32_to_complex32_transform then
        cufft_assert(cufft_c.cufftExecC2C(p.cufft_p, [&cufft_c.cufftComplex](input_base), [&cufft_c.cufftComplex](output_base), cufft_c.CUFFT_FORWARD))
      elseif double_to_complex64_transform then
        cufft_assert(cufft_c.cufftExecD2Z(p.cufft_p, [&cufft_c.cufftDoubleReal](input_base), [&cufft_c.cufftDoubleComplex](output_base)))
      elseif complex64_to_complex64_transform then
        cufft_assert(cufft_c.cufftExecZ2Z(p.cufft_p, [&cufft_c.cufftDoubleComplex](input_base), [&cufft_c.cufftDoubleComplex](output_base), cufft_c.CUFFT_FORWARD))
      end
    end
  end

  --- Execute plan. This is the __inline version of the task should the user wish to use that.
  -- @param input Input to execute plan on.
  -- @param output Output of the executed plan.
  -- @param plan Plan object used to execute plan.
  -- @note For GPU, it calls cufftExecZ2Z.
  -- @note For CPU, it calls fftw_execute_dft.
  __demand(__inline)
  task iface.execute_plan(input : region(ispace(itype), dtype_in),
                          output : region(ispace(itype), dtype_out),
                          plan : region(ispace(int1d), iface.plan))
  where reads writes (input, output, plan) do
    var p = iface.get_plan(plan, true)
    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime()))
    var input_base = get_base_in(rect_in_t(input.ispace.bounds), __physical(input)[0], __fields(input)[0]).base
    var output_base = get_base_out(rect_out_t(output.ispace.bounds), __physical(output)[0], __fields(output)[0]).base

    if float_to_complex32_transform then
      fftw_c.fftwf_execute_dft_r2c(p.float_p, [&float](input_base), [&fftw_c.fftwf_complex](output_base))
    elseif complex32_to_complex32_transform then
      fftw_c.fftwf_execute_dft(p.float_p, [&fftw_c.fftwf_complex](input_base), [&fftw_c.fftwf_complex](output_base))
    elseif double_to_complex64_transform then
      fftw_c.fftw_execute_dft_r2c(p.p, [&double](input_base), [&fftw_c.fftw_complex](output_base))
    elseif complex64_to_complex64_transform then
      fftw_c.fftw_execute_dft(p.p, [&fftw_c.fftw_complex](input_base), [&fftw_c.fftw_complex](output_base))
    end

    p.address_space = address_space

    rescape
      if gpu_available then
        remit rquote
          if iface.get_num_local_gpus() > 0 then
            execute_plan_gpu(input, output, plan, p.address_space)
          end
        end
      end
    end
  end

  --- Execute plan - task version. As execute_plan is a __demand(__inline) task, we provide make_plan_task as a wrapper for convenience, should the user wish to use a new task and not an inlined one.
  -- @param input Input to execute plan on.
  -- @param output Output of the executed plan.
  -- @param plan Plan object used to execute plan.
  task iface.execute_plan_task(input : region(ispace(itype), dtype_in),
                               output : region(ispace(itype), dtype_out),
                              plan : region(ispace(int1d), iface.plan))
  where reads writes (input, output, plan) do
    iface.execute_plan(input, output, plan)
  end

  --- Execute plan (distributed version). This is intended to be called from inside the user's main and avoids the need for the user to directly (say) index launch execute_plan_task repeatedly.
  -- @param input Input region.
  -- @param input_part Input partition.
  -- @param output Output region.
  -- @param output_part Output partition.
  -- @param plan Plan region.
  -- @param plan_part Plan partition.
  __demand(__inline)
  task iface.execute_plan_distrib(input : region(ispace(itype), dtype_in),
                               input_part : partition(disjoint, input, ispace(int1d)),
                               output : region(ispace(itype), dtype_out),
                               output_part : partition(disjoint, output, ispace(int1d)),
                               plan : region(ispace(int1d), iface.plan),
                               plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(input, output, plan) do
    __demand(__index_launch)
    for i in input_part.colors do
      iface.execute_plan_task(input_part[i], output_part[i], plan_part[i])
    end
  end

  -- DESTROY PLAN FUNCTIONS

  -- Destroys the FFT plan - GPU version. Called by destroy_plan if needed.
  local destroy_plan_gpu
  if gpu_available then
    __demand(__cuda, __leaf)
    task destroy_plan_gpu(plan : region(ispace(int1d), iface.plan), address_space : c.legion_address_space_t)
    where reads writes(plan) do

      var p = iface.get_plan(plan, true)
      var proc = get_executing_processor(__runtime())

      if c.legion_processor_kind(proc) == c.TOC_PROC then
        regentlib.assert(c.legion_processor_address_space(proc) == address_space, "destory_plan_gpu must be executed on a processor in the same address space")
        cufft_assert(cufft_c.cufftDestroy(p.cufft_p))
      end
    end
  end

  --- Destroy plan. This is the __inline version of the task should the user wish to use that.
  -- @param plan Plan to be destroyed.
  __demand(__inline)
  task iface.destroy_plan(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    var p = iface.get_plan(plan, true)
    var address_space = c.legion_processor_address_space(get_executing_processor(__runtime()))
    fftw_c.fftw_destroy_plan(p.p)
    fftw_c.fftwf_destroy_plan(p.float_p)
    p.address_space = address_space

    -- Call GPU version if applicable
    rescape
      if gpu_available then
        remit rquote
          if iface.get_num_local_gpus() > 0 then
            destroy_plan_gpu(plan, p.address_space)
          end
        end
      end
    end
  end

  --- Destroy plan task. As destroy_plan is a __demand(__inline) task, we provide make_plan_task as a wrapper for convenience, should the user wish to use a new task and not an inlined one.
  -- @param plan Plan to be destroyed.
  task iface.destroy_plan_task(plan : region(ispace(int1d), iface.plan))
  where reads writes(plan) do
    iface.destroy_plan(plan)
  end

  --- Destroy plan (distributed version). This is intended to be called from inside the user's main and avoids the need for the user to directly (say) index launch destroy_plan_task repeatedly.
  -- @param plan Plan to be destroyed.
  -- @param plan_part Plan partition to be destroyed in `plan`.
  __demand(__inline)
  task iface.destroy_plan_distrib(plan : region(ispace(int1d), iface.plan),
                                  plan_part : partition(disjoint, plan, ispace(int1d)))
  where reads writes(plan) do
    __demand(__index_launch)
    for i in plan_part.colors do
      iface.destroy_plan_task(plan_part[i])
    end
  end

  return iface
end

return fft
