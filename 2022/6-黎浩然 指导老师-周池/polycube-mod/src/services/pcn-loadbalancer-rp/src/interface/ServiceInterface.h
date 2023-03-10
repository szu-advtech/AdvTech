/**
* lbrp API
* lbrp API generated from lbrp.yang
*
* OpenAPI spec version: 1.0.0
*
* NOTE: This class is auto generated by the swagger code generator program.
* https://github.com/polycube-network/swagger-codegen.git
* branch polycube
*/


/* Do not edit this file manually */

/*
* ServiceInterface.h
*
*
*/

#pragma once

#include "../serializer/ServiceJsonObject.h"

#include "../ServiceBackend.h"

using namespace io::swagger::server::model;

class ServiceInterface {
public:

  virtual void update(const ServiceJsonObject &conf) = 0;
  virtual ServiceJsonObject toJsonObject() = 0;

  /// <summary>
  /// Service name related to the backend server of the pool is connected to
  /// </summary>
  virtual std::string getName() = 0;
  virtual void setName(const std::string &value) = 0;

  /// <summary>
  /// Virtual IP (vip) of the service where clients connect to
  /// </summary>
  virtual std::string getVip() = 0;

  /// <summary>
  /// Port of the virtual server where clients connect to (this value is ignored in case of ICMP)
  /// </summary>
  virtual uint16_t getVport() = 0;

  /// <summary>
  /// Upper-layer protocol associated with a loadbalancing service instance. &#39;ALL&#39; creates an entry for all the supported protocols
  /// </summary>
  virtual ServiceProtoEnum getProto() = 0;

  /// <summary>
  /// Pool of backend servers that actually serve requests
  /// </summary>
  virtual std::shared_ptr<ServiceBackend> getBackend(const std::string &ip) = 0;
  virtual std::vector<std::shared_ptr<ServiceBackend>> getBackendList() = 0;
  virtual void addBackend(const std::string &ip, const ServiceBackendJsonObject &conf) = 0;
  virtual void addBackendList(const std::vector<ServiceBackendJsonObject> &conf) = 0;
  virtual void replaceBackend(const std::string &ip, const ServiceBackendJsonObject &conf) = 0;
  virtual void replaceBackendList(const std::vector<ServiceBackendJsonObject> &conf) = 0;
  virtual void delBackend(const std::string &ip) = 0;
  virtual void delBackendList() = 0;
};

