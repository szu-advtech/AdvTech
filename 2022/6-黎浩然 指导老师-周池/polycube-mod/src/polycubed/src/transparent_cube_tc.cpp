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

#include "transparent_cube_tc.h"
#include "cube_tc.h"
#include "datapath_log.h"
#include "exceptions.h"
#include "patchpanel.h"

#include <iostream>

namespace polycube {
namespace polycubed {

TransparentCubeTC::TransparentCubeTC(
    const std::string &name, const std::string &service_name,
    const std::vector<std::string> &ingress_code,
    const std::vector<std::string> &egress_code, LogLevel level,
    const service::attach_cb &attach)
    : TransparentCube(name, service_name, PatchPanel::get_tc_instance(),
                      level, CubeType::TC, attach) {
  TransparentCube::init(ingress_code, egress_code);
}

TransparentCubeTC::~TransparentCubeTC() {
  // it cannot be done in Cube::~Cube() because calls a virtual method
  TransparentCube::uninit();
}

std::string TransparentCubeTC::get_wrapper_code() {
  return TransparentCube::get_wrapper_code() + CubeTC::CUBE_TC_COMMON_WRAPPER +
         CubeTC::CUBETC_HELPERS + TRANSPARENTCUBETC_WRAPPER;
}

void TransparentCubeTC::do_compile(int id, uint16_t next,
                                   bool is_netdev, ProgramType type,
                                   LogLevel level_, ebpf::BPF &bpf,
                                   const std::string &code, int index) {
  std::string all_code(get_wrapper_code() +
                       DatapathLog::get_instance().parse_log(code));

  std::vector<std::string> cflags(cflags_);
  cflags.push_back("-DCUBE_ID=" + std::to_string(id));
  cflags.push_back("-DLOG_LEVEL=LOG_" + logLevelString(level_));
  cflags.push_back(std::string("-DCTXTYPE=") + std::string("__sk_buff"));
  cflags.push_back(std::string("-DNEXT=" + std::to_string(next)));
  cflags.push_back(std::string("-DNEXT_IS_NETDEV=") +
                   std::to_string(int(is_netdev)));
  cflags.push_back(std::string("-DPOLYCUBE_PROGRAM_TYPE=") +
                   std::to_string(static_cast<int>(type)));

  std::lock_guard<std::mutex> guard(bcc_mutex);

  auto init_res = bpf.init(all_code, cflags);
  if (init_res.code() != 0) {
    throw BPFError("failed to init ebpf program: " + init_res.msg());
  }
}

void TransparentCubeTC::compile(ebpf::BPF &bpf, const std::string &code,
                                int index, ProgramType type) {
  uint16_t next;
  bool is_netdev;
  switch (type) {
  case ProgramType::INGRESS:
    next = ingress_next_;
    is_netdev = false;
    break;
  case ProgramType::EGRESS:
    next = egress_next_;
    is_netdev = egress_next_is_netdev_;
    break;
  }
  do_compile(get_id(), next, is_netdev, type, level_, bpf, code, index);
}

int TransparentCubeTC::load(ebpf::BPF &bpf, ProgramType type) {
  return CubeTC::do_load(bpf);
}

void TransparentCubeTC::unload(ebpf::BPF &bpf, ProgramType type) {
  CubeTC::do_unload(bpf);
}

const std::string TransparentCubeTC::TRANSPARENTCUBETC_WRAPPER = R"(
int handle_rx_wrapper(struct CTXTYPE *skb) {
  struct pkt_metadata md = {};
  md.packet_len = skb->len;
  md.traffic_class = skb->mark;
  
  int rc = handle_rx(skb, &md);

  // Save the traffic class for the next program in case it was changed
  // by the current one
  skb->mark = md.traffic_class;

  switch (rc) {
    case RX_DROP:
      return TC_ACT_SHOT;
    case RX_OK:
#if NEXT_IS_NETDEV
      return bpf_redirect(NEXT, 0);
#elif NEXT == 0xffff
      return TC_ACT_OK;
#else
      nodes.call(skb, NEXT);
      return TC_ACT_SHOT;
#endif
  }
  return TC_ACT_SHOT;
}

static __always_inline
int pcn_pkt_controller(struct CTXTYPE *skb, struct pkt_metadata *md,
                       u16 reason) {
  // If the packet is tagged add the tagged in the packet itself, otherwise it
  // will be lost
  if (skb->vlan_present) {
    volatile __u32 vlan_tci = skb->vlan_tci;
    volatile __u32 vlan_proto = skb->vlan_proto;
    bpf_skb_vlan_push(skb, vlan_proto, vlan_tci);
  }

  md->cube_id = CUBE_ID;
  // For transparent cubes in_port is used by the controller to know the
  // direction of the packet
  md->in_port = POLYCUBE_PROGRAM_TYPE;
  md->packet_len = skb->len;
  md->reason = reason;

  return controller_tc.perf_submit_skb(skb, skb->len, md, sizeof(*md));
}
)";

}  // namespace polycubed
}  // namespace polycube
