/**
* router API generated from router.yang
*
* NOTE: This file is auto generated by polycube-codegen
* https://github.com/polycube-network/polycube-codegen
*/


/* Do not edit this file manually */

/*
* RouteJsonObject.h
*
*
*/

#pragma once


#include "JsonObjectBase.h"


namespace polycube {
namespace service {
namespace model {


/// <summary>
///
/// </summary>
class  RouteJsonObject : public JsonObjectBase {
public:
  RouteJsonObject();
  RouteJsonObject(const nlohmann::json &json);
  ~RouteJsonObject() final = default;
  nlohmann::json toJson() const final;


  /// <summary>
  /// Destination network IP
  /// </summary>
  std::string getNetwork() const;
  void setNetwork(std::string value);
  bool networkIsSet() const;

  /// <summary>
  /// Next hop; if destination is local will be shown &#39;local&#39; instead of the ip address
  /// </summary>
  std::string getNexthop() const;
  void setNexthop(std::string value);
  bool nexthopIsSet() const;

  /// <summary>
  /// Outgoing interface
  /// </summary>
  std::string getInterface() const;
  void setInterface(std::string value);
  bool interfaceIsSet() const;
  void unsetInterface();

  /// <summary>
  /// Cost of this route
  /// </summary>
  uint32_t getPathcost() const;
  void setPathcost(uint32_t value);
  bool pathcostIsSet() const;
  void unsetPathcost();

private:
  std::string m_network;
  bool m_networkIsSet;
  std::string m_nexthop;
  bool m_nexthopIsSet;
  std::string m_interface;
  bool m_interfaceIsSet;
  uint32_t m_pathcost;
  bool m_pathcostIsSet;
};

}
}
}

