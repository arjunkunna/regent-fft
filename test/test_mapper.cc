/* Copyright 2024 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "test_mapper.h"

#include "mappers/default_mapper.h"
#include "mappers/logging_wrapper.h"
#include "realm/logging.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

static Logger log_fft_test_mapper("fft_test_mapper");

class FFTTestMapper : public DefaultMapper {
public:
  FFTTestMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name);
  Memory default_policy_select_target_memory(
      MapperContext ctx, Processor target_proc, const RegionRequirement &req,
      MemoryConstraint mc = MemoryConstraint()) override;
};

FFTTestMapper::FFTTestMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) {}

Memory FFTTestMapper::default_policy_select_target_memory(MapperContext ctx,
                                                          Processor target_proc,
                                                          const RegionRequirement &req,
                                                          MemoryConstraint mc) {
  // Use zero copy memory for every processor
  // return Memory::Z_COPY_MEM;
  Memory result = Utilities::MachineQueryInterface::find_memory_kind(machine, target_proc,
                                                                     Memory::Z_COPY_MEM);
  if (result.exists()) {
    return result;
  } else {
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
  }
}

static void create_mappers(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs) {
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    FFTTestMapper *mapper =
        new FFTTestMapper(runtime->get_mapper_runtime(), machine, *it, "fft_test_mapper");
    runtime->replace_default_mapper((new LoggingWrapper(mapper)), *it);
  }
}

void register_mappers() { Runtime::add_registration_callback(create_mappers); }
