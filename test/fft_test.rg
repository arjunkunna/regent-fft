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

import "regent"

local fft = require("fft")
local cmapper = require("test_mapper")
local format = require("std/format")

-- PRINT FUNCTIONS

__demand(__inline, __leaf)
task print_array_double_complex(input : region(ispace(int1d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  for x in input do
    var currComplex = input[x]
    format.println("{} + {}j, ", currComplex.real, currComplex.imag)
  end
  format.println("]")
end

__demand(__inline, __leaf)
task print_array_float_complex(input : region(ispace(int1d), complex32), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  for x in input do
    var currComplex = input[x]
    format.println("{} + {}j, ", currComplex.real, currComplex.imag)
  end
  format.println("]")
end

__demand(__inline, __leaf)
task print_array_double_real(input : region(ispace(int1d), double), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  for x in input do
    var currInt = input[x]
    format.println("{}, ", currInt)
  end
  format.println("]")
end

__demand(__inline, __leaf)
task print_array_float_real(input : region(ispace(int1d), float), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  for x in input do
    var currInt = input[x]
    format.println("{},", currInt)
  end
  format.println("]")
end

task print_array_2d_double_complex(input : region(ispace(int2d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  format.println("Bounds = {}", input.bounds)
  for x in input do
    var currComplex = input[x]
    format.println("index {}: {} + {}j,", x, currComplex.real, currComplex.imag)
  end
  format.println("]")
end

task print_array_3d_double_complex(input : region(ispace(int3d), complex64), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  format.println("Bounds = {}", input.bounds)
  for x in input do
    var currComplex = input[x]
    format.println("index {}: {} + {}j,", x, currComplex.real, currComplex.imag)
  end
  format.println("]")
end

task print_array_3d_double_real(input : region(ispace(int3d), double), arrayName: rawstring)
where reads (input) do
  format.println("{} = [", arrayName)
  format.println("Bounds = {}", input.bounds)
  for x in input do
    var currInt = input[x]
    format.println("{},", currInt)
  end
  format.println("]")
end

-- task compare_regions_double_complex(r1 : region(ispace(int1d), complex64), r2 : region(ispace(int1d), complex64))
-- where reads (r1, r2) do
--  var regions_same = true
--  for x in r1 do
--    if (r1[x].real == r2[x].real) and (r1[x].imag == r2[x].imag) then
--      format.println("indices match")
--    else
--      format.println("Regions are the not the same: r1 real is {}, r2 real is {}, r1 imag is {}, r2 imag is {} for index {}", r1[x].real, r2[x].real, r1[x].imag, r2[x].imag, x)
--      regions_same = false
--    end
--  end
--  if regions_same == true then
--    format.println("Regions are the same")
--  end
--  return regions_same
-- end

-- INTERFACES

local fft1d_complex32_complex32 = fft.generate_fft_interface(int1d, complex32, complex32)

local fft1d_complex64_complex64 = fft.generate_fft_interface(int1d, complex64, complex64)
local fft2d_complex64_complex64 = fft.generate_fft_interface(int2d, complex64, complex64)
local fft3d_complex64_complex64 = fft.generate_fft_interface(int3d, complex64, complex64)

local fft1d_double_complex64 = fft.generate_fft_interface(int1d, double, complex64)
local fft1d_float_complex32 = fft.generate_fft_interface(int1d, float, complex32)
local fft3d_batch_complex64_complex64 = fft.generate_fft_interface(int3d, complex64, complex64)
local fft3d_batch_double_complex64 = fft.generate_fft_interface(int3d, double, complex64)

-- TESTS

__demand(__inline)
task test_fft1d_double_to_complex64_transform()
  var r = region(ispace(int1d, 3), double)
  var s = region(ispace(int1d, 3), complex64)
  var p = region(ispace(int1d, 1), fft1d_double_complex64.plan)

  fill(r, 3)
  fill(s, 0)
  print_array_double_real(r, "Input array")

  fft1d_double_complex64.make_plan(r, s, p)
  fft1d_double_complex64.execute_plan_task(r, s, p)
  print_array_double_complex(s, "Output array")
  fft1d_double_complex64.destroy_plan(p)
end

__demand(__inline)
task test_fft1d_float_to_complex32_transform()
  var r = region(ispace(int1d, 3), float)
  var s = region(ispace(int1d, 3), complex32)
  var p = region(ispace(int1d, 1), fft1d_float_complex32.plan)

  fill(r, 3)
  fill(s, 0)
  print_array_float_real(r, "Input array")

  fft1d_float_complex32.make_plan(r, s, p)
  fft1d_float_complex32.execute_plan_task(r, s, p)
  print_array_float_complex(s, "Output array")
  fft1d_float_complex32.destroy_plan(p)
end

__demand(__inline)
task test_fft1d_complex32_to_complex32_transform()
  var r = region(ispace(int1d, 3), complex32)
  var s = region(ispace(int1d, 3), complex32)
  var p = region(ispace(int1d, 1), fft1d_complex32_complex32.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end

  fill(s, 0)
  print_array_float_complex(r, "Input array")

  fft1d_complex32_complex32.make_plan(r, s, p)
  fft1d_complex32_complex32.execute_plan_task(r, s, p)
  print_array_float_complex(s, "Output array")
  fft1d_complex32_complex32.destroy_plan(p)
end

__demand(__inline)
task test_fft1d_complex64_to_complex64_transform()
  var r = region(ispace(int1d, 5), complex64)
  var s = region(ispace(int1d, 5), complex64)
  var p = region(ispace(int1d, 1), fft1d_complex64_complex64.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end

  fill(s, 0)
  print_array_double_complex(r, "Input array")

  fft1d_complex64_complex64.make_plan(r, s, p)
  fft1d_complex64_complex64.execute_plan_task(r, s, p)
  print_array_double_complex(s, "Output array")
  fft1d_complex64_complex64.destroy_plan(p)
end

__demand(__inline)
task test_fft1d_complex64_to_complex64_distrib_transform()
  var n = fft1d_complex64_complex64.get_num_nodes()
  format.println("Num nodes in distrib is {}", n)

  var r = region(ispace(int1d, 3*n), complex64)
  var r_part = partition(equal, r, ispace(int1d, n))
  var s = region(ispace(int1d, 3*n), complex64)
  var s_part = partition(equal, s, ispace(int1d, n))
  var p = region(ispace(int1d, n), fft1d_complex64_complex64.plan)
  var p_part = partition(equal, p, ispace(int1d, n))

  for x in r do
    r[x].real = 4
    r[x].imag = 4
  end
  fill(s, 0)
  print_array_double_complex(r, "Input array for distrib")
  -- Important: this overwrites r and s!
  fft1d_complex64_complex64.make_plan_distrib(r, r_part, s, s_part, p, p_part)
  __demand(__index_launch)
  for i in r_part.colors do
    fft1d_complex64_complex64.execute_plan_task(r_part[i], s_part[i], p)
  end
  print_array_double_complex(s, "Output array for distrib")
  fft1d_complex64_complex64.destroy_plan_distrib(p, p_part)
end

__demand(__inline)
task test_fft2d_complex64_to_complex64_transform()
  var r = region(ispace(int2d, { 2, 2 }), complex64)
  var s = region(ispace(int2d, { 2, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft2d_complex64_complex64.plan)

  for x in r do
    r[x].real = 5
    r[x].imag = 5
  end
  fill(s, 1)

  print_array_2d_double_complex(r, "Input array")
  fft2d_complex64_complex64.make_plan(r, s, p)
  fft2d_complex64_complex64.execute_plan_task(r, s, p)
  print_array_2d_double_complex(s, "Output array")
  fft2d_complex64_complex64.destroy_plan(p)
end

__demand(__inline)
task test_fft3d_complex64_to_complex64_transform()
  var r = region(ispace(int3d, { 3, 2, 2 }), complex64)
  var s = region(ispace(int3d, { 3, 2, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft3d_complex64_complex64.plan)
  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)
  print_array_3d_double_complex(r, "Input array")
  -- Important: this overwrites r and s!
  fft3d_complex64_complex64.make_plan(r, s, p)
  fft3d_complex64_complex64.execute_plan_task(r, s, p)
  print_array_3d_double_complex(s, "Output array")
  fft3d_complex64_complex64.destroy_plan(p)
end

__demand(__inline)
task test_fft3d_complex64_to_complex64_batch_transform()
  var r = region(ispace(int3d, { 3, 3, 2 }), complex64)
  var s = region(ispace(int3d, { 3, 3, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft3d_batch_complex64_complex64.plan)

  for x in r do
    r[x].real = 3
    r[x].imag = 3
  end
  fill(s, 0)

  print_array_3d_double_complex(r, "Input array")
  fft3d_batch_complex64_complex64.make_plan_batch(r, s, p)
  fft3d_batch_complex64_complex64.execute_plan_task(r, s, p)
  print_array_3d_double_complex(s, "Output array")
  fft3d_batch_complex64_complex64.destroy_plan(p)
end

__demand(__inline)
task test_fft3d_double_to_complex64_batch_transform()
  var r = region(ispace(int3d, { 3, 3, 2 }), double)
  var s = region(ispace(int3d, { 3, 3, 2 }), complex64)
  var p = region(ispace(int1d, 1), fft3d_batch_double_complex64.plan)

  -- fill(r, 3)
  for x in r do
    r[x] = 3
  end
  fill(s, 0)

  print_array_3d_double_real(r, "Input array")
  fft3d_batch_double_complex64.make_plan_batch(r, s, p)
  fft3d_batch_double_complex64.execute_plan_task(r, s, p)
  print_array_3d_double_complex(s, "Output array")
  fft3d_batch_double_complex64.destroy_plan(p)
end

task main()
  test_fft1d_double_to_complex64_transform()
  test_fft1d_complex32_to_complex32_transform()
  test_fft1d_float_to_complex32_transform()
  test_fft1d_complex64_to_complex64_transform()
  test_fft1d_complex64_to_complex64_distrib_transform()
  test_fft2d_complex64_to_complex64_transform()
  test_fft3d_complex64_to_complex64_transform()
  test_fft3d_complex64_to_complex64_batch_transform()
  test_fft3d_double_to_complex64_batch_transform()
end

regentlib.start(main, cmapper.register_mappers)
