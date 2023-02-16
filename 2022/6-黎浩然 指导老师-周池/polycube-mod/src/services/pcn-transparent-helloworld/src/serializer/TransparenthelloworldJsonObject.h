/**
* transparenthelloworld API
* transparenthelloworld API generated from transparenthelloworld.yang
*
* OpenAPI spec version: 1.0.0
*
* NOTE: This class is auto generated by the swagger code generator program.
* https://github.com/polycube-network/swagger-codegen.git
* branch polycube
*/


/* Do not edit this file manually */

/*
* TransparenthelloworldJsonObject.h
*
*
*/

#pragma once


#include "JsonObjectBase.h"

#include "polycube/services/cube.h"

namespace io {
namespace swagger {
namespace server {
namespace model {

enum class TransparenthelloworldIngressActionEnum {
  DROP, PASS, SLOWPATH
};
enum class TransparenthelloworldEgressActionEnum {
  DROP, PASS, SLOWPATH
};

/// <summary>
///
/// </summary>
class  TransparenthelloworldJsonObject : public JsonObjectBase {
public:
  TransparenthelloworldJsonObject();
  TransparenthelloworldJsonObject(const nlohmann::json &json);
  ~TransparenthelloworldJsonObject() final = default;
  nlohmann::json toJson() const final;


  /// <summary>
  /// Name of the transparenthelloworld service
  /// </summary>
  std::string getName() const;
  void setName(std::string value);
  bool nameIsSet() const;

  /// <summary>
  /// Action performed on ingress packets
  /// </summary>
  TransparenthelloworldIngressActionEnum getIngressAction() const;
  void setIngressAction(TransparenthelloworldIngressActionEnum value);
  bool ingressActionIsSet() const;
  void unsetIngressAction();
  static std::string TransparenthelloworldIngressActionEnum_to_string(const TransparenthelloworldIngressActionEnum &value);
  static TransparenthelloworldIngressActionEnum string_to_TransparenthelloworldIngressActionEnum(const std::string &str);

  /// <summary>
  /// Action performed on egress packets
  /// </summary>
  TransparenthelloworldEgressActionEnum getEgressAction() const;
  void setEgressAction(TransparenthelloworldEgressActionEnum value);
  bool egressActionIsSet() const;
  void unsetEgressAction();
  static std::string TransparenthelloworldEgressActionEnum_to_string(const TransparenthelloworldEgressActionEnum &value);
  static TransparenthelloworldEgressActionEnum string_to_TransparenthelloworldEgressActionEnum(const std::string &str);

private:
  std::string m_name;
  bool m_nameIsSet;
  TransparenthelloworldIngressActionEnum m_ingressAction;
  bool m_ingressActionIsSet;
  TransparenthelloworldEgressActionEnum m_egressAction;
  bool m_egressActionIsSet;
};

}
}
}
}

