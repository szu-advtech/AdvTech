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

#include "../Firewall.h"
#include "datapaths/Firewall_ActionLookup_dp.h"
#include "polycube/common.h"

Firewall::ActionLookup::ActionLookup(const int &index,
                                     const ChainNameEnum &direction,
                                     Firewall &outer)
    : Firewall::Program(firewall_code_actionlookup, index, direction, outer) {

  load();
}

Firewall::ActionLookup::~ActionLookup() {}

std::string Firewall::ActionLookup::getCode() {
  std::string noMacroCode = code;

  /*Replacing the maximum number of rules*/
  replaceAll(noMacroCode, "_MAXRULES", std::to_string(FROM_NRULES_TO_NELEMENTS(firewall.maxRules)));

  /*Replacing nrElements*/
  replaceAll(noMacroCode, "_NR_ELEMENTS",
             std::to_string(FROM_NRULES_TO_NELEMENTS(
                 firewall.getChain(direction)->getNrRules())));

  /*Pointing to the module in charge of updating the conn table and forwarding*/
  replaceAll(noMacroCode, "_CONNTRACKTABLEUPDATE",
             std::to_string(ModulesConstants::CONNTRACKTABLEUPDATE));

  return noMacroCode;
}

uint64_t Firewall::ActionLookup::getPktsCount(int ruleNumber) {
  std::string tableName = "pktsCounter";
  try {
    uint64_t pkts = 0;
    auto pktsTable = firewall.get_percpuarray_table<uint64_t>(tableName, index,
                                                              getProgramType());
    auto values = pktsTable.get(ruleNumber);

    return std::accumulate(values.begin(), values.end(), pkts);
  } catch (...) {
    throw std::runtime_error("Counter not available.");
  }
}

uint64_t Firewall::ActionLookup::getBytesCount(int ruleNumber) {
  std::string tableName = "bytesCounter";

  try {
    uint64_t bytes = 0;
    auto bytesTable =
        firewall.get_percpuarray_table<uint64_t>(tableName, index,
                                                 getProgramType());
    auto values = bytesTable.get(ruleNumber);

    return std::accumulate(values.begin(), values.end(), bytes);
  } catch (...) {
    throw std::runtime_error("Counter not available.");
  }
}

void Firewall::ActionLookup::flushCounters(int ruleNumber) {
  std::string pktsTableName = "pktsCounter";
  std::string bytesTableName = "bytesCounter";

  try {
    auto pktsTable =
        firewall.get_percpuarray_table<uint64_t>(pktsTableName, index,
                                                 getProgramType());
    auto bytesTable =
        firewall.get_percpuarray_table<uint64_t>(bytesTableName, index,
                                                 getProgramType());

    pktsTable.set(ruleNumber, 0);
    bytesTable.set(ruleNumber, 0);
  } catch (std::exception &e) {
    throw std::runtime_error("Counters not available: " +
                             std::string(e.what()));
  }
}

bool Firewall::ActionLookup::updateTableValue(int ruleNumber, int action) {
  std::string tableName = "actions";
  try {
    auto actionsTable = firewall.get_array_table<int>(tableName, index,
                                                      getProgramType());
    actionsTable.set(ruleNumber, action);
  } catch (...) {
    return false;
  }
  return true;
}
