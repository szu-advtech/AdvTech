/**
* router API generated from router.yang
*
* NOTE: This file is auto generated by polycube-codegen
* https://github.com/polycube-network/polycube-codegen
*/


/* Do not edit this file manually */

/*
* ArpTableBase.h
*
*
*/

#pragma once

#include "../serializer/ArpTableJsonObject.h"






#include <spdlog/spdlog.h>

using namespace polycube::service::model;

class Router;

class ArpTableBase {
 public:
  
  ArpTableBase(Router &parent);
  
  virtual ~ArpTableBase();
  virtual void update(const ArpTableJsonObject &conf);
  virtual ArpTableJsonObject toJsonObject();

  /// <summary>
  /// Destination IP address
  /// </summary>
  virtual std::string getAddress() = 0;

  /// <summary>
  /// Destination MAC address
  /// </summary>
  virtual std::string getMac() = 0;
  virtual void setMac(const std::string &value) = 0;

  /// <summary>
  /// Outgoing interface
  /// </summary>
  virtual std::string getInterface() = 0;
  virtual void setInterface(const std::string &value) = 0;

  std::shared_ptr<spdlog::logger> logger();
 protected:
  Router &parent_;
};
