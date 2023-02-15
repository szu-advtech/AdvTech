/*
 * Copyright 2017 The Polycube Authors
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

#include "patchpanel.h"

#include <iostream>

namespace polycube {
namespace polycubed {

const std::string PatchPanel::PATCHPANEL_CODE = R"(
BPF_TABLE_PUBLIC("prog", int, int, _MAP_NAME, _POLYCUBE_MAX_NODES);
)";

std::set<uint16_t> PatchPanel::node_ids_ = std::set<uint16_t>();

std::array<bool, PatchPanel::_POLYCUBE_MAX_NODES>
    PatchPanel::nodes_present_tc_ = {false};

std::array<bool, PatchPanel::_POLYCUBE_MAX_NODES>
    PatchPanel::nodes_present_xdp_ = {false};

PatchPanel &PatchPanel::get_tc_instance() {
  static PatchPanel tc_instance("nodes", nodes_present_tc_);
  return tc_instance;
}

PatchPanel &PatchPanel::get_xdp_instance() {
  static PatchPanel xdp_instance("xdp_nodes", nodes_present_xdp_);
  return xdp_instance;
}

PatchPanel::PatchPanel(const std::string &map_name,
                       std::array<bool, _POLYCUBE_MAX_NODES> &nodes_present)
    : logger(spdlog::get("polycubed")), nodes_present_(nodes_present) {
  static bool initialized = false;

  std::vector<std::string> flags;
  // flags.push_back(std::string("-DMAP_NAME=") + map_name);
  flags.push_back(std::string("-D_POLYCUBE_MAX_NODES=") +
                  std::to_string(_POLYCUBE_MAX_NODES));
  std::string code(PATCHPANEL_CODE);
  code.replace(code.find("_MAP_NAME"), 9, map_name);

  auto init_res = program_.init(code, flags);
  if (init_res.code() != 0) {
    logger->error("error creating patch panel: {0}", init_res.msg());
    throw std::runtime_error("Error creating patch panel");
  }

  if (!initialized) {
    for (uint16_t i = 1; i < _POLYCUBE_MAX_NODES; i++) {
      node_ids_.insert(i);
    }

    initialized = true;
  }

  // TODO: is this code valid?
  // (implicit copy constructor should be ok for this case)
  auto a = program_.get_prog_table(map_name);
  nodes_ = std::unique_ptr<ebpf::BPFProgTable>(new ebpf::BPFProgTable(a));
}

PatchPanel::~PatchPanel() {}

uint16_t PatchPanel::add(int fd) {
  int p = *node_ids_.begin();
  node_ids_.erase(p);
  nodes_->update_value(p, fd);
  nodes_present_[p] = true;
  return p;
}

void PatchPanel::add(int fd, uint16_t index) {
  if (nodes_present_[index]) {
    logger->error("Index '{0}' is busy in patch panel", index);
    throw std::runtime_error("Index is busy");
  }

  node_ids_.erase(index);
  nodes_->update_value(index, fd);
  nodes_present_[index] = true;
}

void PatchPanel::remove(uint16_t index) {
  nodes_->remove_value(index);
  nodes_present_tc_[index] = false;
  nodes_present_xdp_[index] = false;
  node_ids_.insert(index);
}

void PatchPanel::update(uint16_t index, int fd) {
  if (!nodes_present_[index]) {
    logger->error("Index '{0}' is not registered", index);
    throw std::runtime_error("Index is not registered");
  }

  nodes_->update_value(index, fd);
}

}  // namespace polycubed
}  // namespace polycube
