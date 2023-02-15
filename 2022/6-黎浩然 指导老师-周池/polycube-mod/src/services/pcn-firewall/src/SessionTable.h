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

#pragma once

#include "base/SessionTableBase.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <fstream>

#define UDP_ESTABLISHED_TIMEOUT 180000000000
#define UDP_NEW_TIMEOUT 30000000000
#define ICMP_TIMEOUT 30000000000
#define TCP_ESTABLISHED 432000000000000
#define TCP_SYN_SENT 120000000000
#define TCP_SYN_RECV 60000000000
#define TCP_LAST_ACK 30000000000
#define TCP_FIN_WAIT 120000000000
#define TCP_TIME_WAIT 120000000000

enum {
  NEW,
  ESTABLISHED,
  RELATED,
  INVALID,
  SYN_SENT,
  SYN_RECV,
  FIN_WAIT_1,
  FIN_WAIT_2,
  LAST_ACK,
  TIME_WAIT
};

class Firewall;

//using namespace io::swagger::server::model;
using namespace polycube::service::model;


class SessionTable : public SessionTableBase {
 public:
  SessionTable(Firewall &parent, const SessionTableJsonObject &conf);
  virtual ~SessionTable();

  std::shared_ptr<spdlog::logger> logger();
  void update(const SessionTableJsonObject &conf) override;
  SessionTableJsonObject toJsonObject() override;

  /// <summary>
  /// Source IP
  /// </summary>
  std::string getSrc() override;

  /// <summary>
  /// Destination IP
  /// </summary>
  std::string getDst() override;

  /// <summary>
  /// Connection state.
  /// </summary>
  std::string getState() override;

  /// <summary>
  /// Last packet matching the connection
  /// </summary>
  uint32_t getEta() override;

  /// <summary>
  /// Level 4 Protocol.
  /// </summary>
  std::string getL4proto() override;

  /// <summary>
  /// Destination
  /// </summary>
  uint16_t getDport() override;

  /// <summary>
  /// Source Port
  /// </summary>
  uint16_t getSport() override;

  static std::string state_from_number_to_string(int state);
  static uint32_t from_ttl_to_eta(uint64_t ttl, uint16_t state,
                                  uint16_t l4proto);
  //static uint64_t hex_string_to_uint64(const std::string &str);

 private:
  SessionTableJsonObject fields;
};
