-- Copyright 2020 Stanford University
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

local fft1d = fft.generate_fft_interface(int1d, complex64)
local fft2d = fft.generate_fft_interface(int2d, complex64)

__demand(__inline)
task test1d()
  var r = region(ispace(int1d, 128), complex64)
  var s = region(ispace(int1d, 128), complex64)
  fill(r, 0)
  fill(s, 0)
  var p = fft1d.make_plan(r, s)
  fill(r, 0)
  fill(s, 0)
  fft1d.execute_plan(r, s, p)
  fft1d.destroy_plan(p)
end

__demand(__inline)
task test2d()
  var r = region(ispace(int2d, { 128, 128 }), complex64)
  var s = region(ispace(int2d, { 128, 128 }), complex64)
  fill(r, 0)
  fill(s, 0)
  var p = fft2d.make_plan(r, s)
  fill(r, 0)
  fill(s, 0)
  fft2d.execute_plan(r, s, p)
  fft2d.destroy_plan(p)
end

task main()
  test1d()
  test2d()
end
regentlib.start(main)
