/**
* ddosmitigator API
* ddosmitigator API generated from ddosmitigator.yang
*
* OpenAPI spec version: 1.0.0
*
* NOTE: This class is auto generated by the swagger code generator program.
* https://github.com/polycube-network/swagger-codegen.git
* branch polycube
*/


/* Do not edit this file manually */



#include "BlacklistSrcJsonObject.h"
#include <regex>

namespace io {
namespace swagger {
namespace server {
namespace model {

BlacklistSrcJsonObject::BlacklistSrcJsonObject() {
  m_ipIsSet = false;
  m_dropPktsIsSet = false;
}

BlacklistSrcJsonObject::BlacklistSrcJsonObject(const nlohmann::json &val) :
  JsonObjectBase(val) {
  m_ipIsSet = false;
  m_dropPktsIsSet = false;


  if (val.count("ip")) {
    setIp(val.at("ip").get<std::string>());
  }

  if (val.count("drop-pkts")) {
    setDropPkts(val.at("drop-pkts").get<uint64_t>());
  }
}

nlohmann::json BlacklistSrcJsonObject::toJson() const {
  nlohmann::json val = nlohmann::json::object();
  if (!getBase().is_null()) {
    val.update(getBase());
  }

  if (m_ipIsSet) {
    val["ip"] = m_ip;
  }

  if (m_dropPktsIsSet) {
    val["drop-pkts"] = m_dropPkts;
  }

  return val;
}

std::string BlacklistSrcJsonObject::getIp() const {
  return m_ip;
}

void BlacklistSrcJsonObject::setIp(std::string value) {
  m_ip = value;
  m_ipIsSet = true;
}

bool BlacklistSrcJsonObject::ipIsSet() const {
  return m_ipIsSet;
}



uint64_t BlacklistSrcJsonObject::getDropPkts() const {
  return m_dropPkts;
}

void BlacklistSrcJsonObject::setDropPkts(uint64_t value) {
  m_dropPkts = value;
  m_dropPktsIsSet = true;
}

bool BlacklistSrcJsonObject::dropPktsIsSet() const {
  return m_dropPktsIsSet;
}

void BlacklistSrcJsonObject::unsetDropPkts() {
  m_dropPktsIsSet = false;
}


}
}
}
}

