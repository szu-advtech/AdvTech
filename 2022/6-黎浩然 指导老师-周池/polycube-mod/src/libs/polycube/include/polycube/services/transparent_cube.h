/*
 * Copyright 2018 The Polycube Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <map>
#include <string>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include "polycube/common.h"

#include "polycube/services/cube_factory.h"
#include "polycube/services/utils.h"

#include "polycube/services/base_cube.h"

namespace Tins {
class EthernetII;
}

using Tins::EthernetII;

namespace polycube {
namespace service {

class TransparentCube : public BaseCube {
 public:
  TransparentCube(const nlohmann::json &conf,
                  const std::vector<std::string> &ingress_code,
                  const std::vector<std::string> &egress_code);
  virtual ~TransparentCube();

  virtual void packet_in(Direction direction, PacketInMetadata &md,
                         const std::vector<uint8_t> &packet) = 0;

  void send_packet_out(EthernetII &packet, Direction direction,
                       bool recirculate = false);

  void set_conf(const nlohmann::json &conf);
  nlohmann::json to_json() const;

  virtual void attach();

  void subscribe_parent_parameter(const std::string &param_name,
                                  ParameterEventCallback &callback);
  void unsubscribe_parent_parameter(const std::string &param_name);
  std::string get_parent_parameter(const std::string &param_name);

 private:
  std::shared_ptr<TransparentCubeIface>
      cube_;  // pointer to the cube in polycubed
  packet_in_cb handle_packet_in;
};

}  // namespace service
}  // namespace polycube
