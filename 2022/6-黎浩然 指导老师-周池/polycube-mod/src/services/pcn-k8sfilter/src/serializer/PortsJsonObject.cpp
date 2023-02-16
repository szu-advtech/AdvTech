/**
* k8sfilter API
* k8sfilter API generated from k8sfilter.yang
*
* OpenAPI spec version: 1.0.0
*
* NOTE: This class is auto generated by the swagger code generator program.
* https://github.com/polycube-network/swagger-codegen.git
* branch polycube
*/


/* Do not edit this file manually */



#include "PortsJsonObject.h"
#include <regex>

namespace io {
namespace swagger {
namespace server {
namespace model {

PortsJsonObject::PortsJsonObject() {
  m_nameIsSet = false;
  m_typeIsSet = false;
}

PortsJsonObject::PortsJsonObject(const nlohmann::json &val) :
  JsonObjectBase(val) {
  m_nameIsSet = false;
  m_typeIsSet = false;


  if (val.count("name")) {
    setName(val.at("name").get<std::string>());
  }

  if (val.count("type")) {
    setType(string_to_PortsTypeEnum(val.at("type").get<std::string>()));
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
    case PortsTypeEnum::EXTERNAL:
      return std::string("external");
    case PortsTypeEnum::INTERNAL:
      return std::string("internal");
    default:
      throw std::runtime_error("Bad Ports type");
  }
}

PortsTypeEnum PortsJsonObject::string_to_PortsTypeEnum(const std::string &str){
  if (JsonObjectBase::iequals("external", str))
    return PortsTypeEnum::EXTERNAL;
  if (JsonObjectBase::iequals("internal", str))
    return PortsTypeEnum::INTERNAL;
  throw std::runtime_error("Ports type is invalid");
}

}
}
}
}

