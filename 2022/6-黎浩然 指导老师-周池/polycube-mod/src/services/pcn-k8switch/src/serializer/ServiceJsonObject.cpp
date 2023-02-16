/**
* k8switch API
* k8switch API generated from k8switch.yang
*
* OpenAPI spec version: 1.0.0
*
* NOTE: This class is auto generated by the swagger code generator program.
* https://github.com/polycube-network/swagger-codegen.git
* branch polycube
*/


/* Do not edit this file manually */



#include "ServiceJsonObject.h"
#include <regex>

namespace io {
namespace swagger {
namespace server {
namespace model {

ServiceJsonObject::ServiceJsonObject() {
  m_nameIsSet = false;
  m_vipIsSet = false;
  m_vportIsSet = false;
  m_protoIsSet = false;
  m_backendIsSet = false;
}

ServiceJsonObject::ServiceJsonObject(const nlohmann::json &val) :
  JsonObjectBase(val) {
  m_nameIsSet = false;
  m_vipIsSet = false;
  m_vportIsSet = false;
  m_protoIsSet = false;
  m_backendIsSet = false;


  if (val.count("name")) {
    setName(val.at("name").get<std::string>());
  }

  if (val.count("vip")) {
    setVip(val.at("vip").get<std::string>());
  }

  if (val.count("vport")) {
    setVport(val.at("vport").get<uint16_t>());
  }

  if (val.count("proto")) {
    setProto(string_to_ServiceProtoEnum(val.at("proto").get<std::string>()));
  }

  if (val.count("backend")) {
    for (auto& item : val["backend"]) {
      ServiceBackendJsonObject newItem{ item };
      m_backend.push_back(newItem);
    }

    m_backendIsSet = true;
  }
}

nlohmann::json ServiceJsonObject::toJson() const {
  nlohmann::json val = nlohmann::json::object();
  if (!getBase().is_null()) {
    val.update(getBase());
  }

  if (m_nameIsSet) {
    val["name"] = m_name;
  }

  if (m_vipIsSet) {
    val["vip"] = m_vip;
  }

  if (m_vportIsSet) {
    val["vport"] = m_vport;
  }

  if (m_protoIsSet) {
    val["proto"] = ServiceProtoEnum_to_string(m_proto);
  }

  {
    nlohmann::json jsonArray;
    for (auto& item : m_backend) {
      jsonArray.push_back(JsonObjectBase::toJson(item));
    }

    if (jsonArray.size() > 0) {
      val["backend"] = jsonArray;
    }
  }

  return val;
}

std::string ServiceJsonObject::getName() const {
  return m_name;
}

void ServiceJsonObject::setName(std::string value) {
  m_name = value;
  m_nameIsSet = true;
}

bool ServiceJsonObject::nameIsSet() const {
  return m_nameIsSet;
}

void ServiceJsonObject::unsetName() {
  m_nameIsSet = false;
}

std::string ServiceJsonObject::getVip() const {
  return m_vip;
}

void ServiceJsonObject::setVip(std::string value) {
  m_vip = value;
  m_vipIsSet = true;
}

bool ServiceJsonObject::vipIsSet() const {
  return m_vipIsSet;
}



uint16_t ServiceJsonObject::getVport() const {
  return m_vport;
}

void ServiceJsonObject::setVport(uint16_t value) {
  m_vport = value;
  m_vportIsSet = true;
}

bool ServiceJsonObject::vportIsSet() const {
  return m_vportIsSet;
}



ServiceProtoEnum ServiceJsonObject::getProto() const {
  return m_proto;
}

void ServiceJsonObject::setProto(ServiceProtoEnum value) {
  m_proto = value;
  m_protoIsSet = true;
}

bool ServiceJsonObject::protoIsSet() const {
  return m_protoIsSet;
}



std::string ServiceJsonObject::ServiceProtoEnum_to_string(const ServiceProtoEnum &value){
  switch(value) {
    case ServiceProtoEnum::TCP:
      return std::string("tcp");
    case ServiceProtoEnum::UDP:
      return std::string("udp");
    default:
      throw std::runtime_error("Bad Service proto");
  }
}

ServiceProtoEnum ServiceJsonObject::string_to_ServiceProtoEnum(const std::string &str){
  if (JsonObjectBase::iequals("tcp", str))
    return ServiceProtoEnum::TCP;
  if (JsonObjectBase::iequals("udp", str))
    return ServiceProtoEnum::UDP;
  throw std::runtime_error("Service proto is invalid");
}
const std::vector<ServiceBackendJsonObject>& ServiceJsonObject::getBackend() const{
  return m_backend;
}

void ServiceJsonObject::addServiceBackend(ServiceBackendJsonObject value) {
  m_backend.push_back(value);
  m_backendIsSet = true;
}


bool ServiceJsonObject::backendIsSet() const {
  return m_backendIsSet;
}

void ServiceJsonObject::unsetBackend() {
  m_backendIsSet = false;
}


}
}
}
}

