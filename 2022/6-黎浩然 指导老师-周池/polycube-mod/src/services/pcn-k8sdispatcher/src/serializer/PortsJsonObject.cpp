/**
* k8sdispatcher API generated from k8sdispatcher.yang
*
* NOTE: This file is auto generated by polycube-codegen
* https://github.com/polycube-network/polycube-codegen
*/


/* Do not edit this file manually */



#include "PortsJsonObject.h"
#include <regex>

namespace polycube {
namespace service {
namespace model {

PortsJsonObject::PortsJsonObject() {
  m_nameIsSet = false;
  m_typeIsSet = false;
  m_ipIsSet = false;
}

PortsJsonObject::PortsJsonObject(const nlohmann::json &val) :
  JsonObjectBase(val) {
  m_nameIsSet = false;
  m_typeIsSet = false;
  m_ipIsSet = false;


  if (val.count("name")) {
    setName(val.at("name").get<std::string>());
  }

  if (val.count("type")) {
    setType(string_to_PortsTypeEnum(val.at("type").get<std::string>()));
  }

  if (val.count("ip")) {
    setIp(val.at("ip").get<std::string>());
  }
}

nlohmann::json PortsJsonObject::toJson() const {
  nlohmann::json val = nlohmann::json::object();
  if (!getBase().is_null()) {
    val.update(getBase());
  }

  if (m_nameIsSet) {
    val["name"] = m_name;
  }

  if (m_typeIsSet) {
    val["type"] = PortsTypeEnum_to_string(m_type);
  }

  if (m_ipIsSet) {
    val["ip"] = m_ip;
  }

  return val;
}

std::string PortsJsonObject::getName() const {
  return m_name;
}

void PortsJsonObject::setName(std::string value) {
  m_name = value;
  m_nameIsSet = true;
}

bool PortsJsonObject::nameIsSet() const {
  return m_nameIsSet;
}



PortsTypeEnum PortsJsonObject::getType() const {
  return m_type;
}

void PortsJsonObject::setType(PortsTypeEnum value) {
  m_type = value;
  m_typeIsSet = true;
}

bool PortsJsonObject::typeIsSet() const {
  return m_typeIsSet;
}



std::string PortsJsonObject::PortsTypeEnum_to_string(const PortsTypeEnum &value){
  switch(value) {
    case PortsTypeEnum::BACKEND:
      return std::string("backend");
    case PortsTypeEnum::FRONTEND:
      return std::string("frontend");
    default:
      throw std::runtime_error("Bad Ports type");
  }
}

PortsTypeEnum PortsJsonObject::string_to_PortsTypeEnum(const std::string &str){
  if (JsonObjectBase::iequals("backend", str))
    return PortsTypeEnum::BACKEND;
  if (JsonObjectBase::iequals("frontend", str))
    return PortsTypeEnum::FRONTEND;
  throw std::runtime_error("Ports type is invalid");
}
std::string PortsJsonObject::getIp() const {
  return m_ip;
}

void PortsJsonObject::setIp(std::string value) {
  m_ip = value;
  m_ipIsSet = true;
}

bool PortsJsonObject::ipIsSet() const {
  return m_ipIsSet;
}

void PortsJsonObject::unsetIp() {
  m_ipIsSet = false;
}


}
}
}

