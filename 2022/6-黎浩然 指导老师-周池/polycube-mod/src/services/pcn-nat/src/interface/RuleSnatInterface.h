/**
* nat API
* nat API generated from nat.yang
*
* OpenAPI spec version: 1.0.0
*
* NOTE: This class is auto generated by the swagger code generator program.
* https://github.com/polycube-network/swagger-codegen.git
* branch polycube
*/


/* Do not edit this file manually */

/*
* RuleSnatInterface.h
*
*
*/

#pragma once

#include "../serializer/RuleSnatJsonObject.h"
#include "../serializer/RuleSnatAppendOutputJsonObject.h"
#include "../serializer/RuleSnatAppendInputJsonObject.h"

#include "../RuleSnatEntry.h"

using namespace io::swagger::server::model;

class RuleSnatInterface {
public:

  virtual void update(const RuleSnatJsonObject &conf) = 0;
  virtual RuleSnatJsonObject toJsonObject() = 0;

  /// <summary>
  /// List of Source NAT rules
  /// </summary>
  virtual std::shared_ptr<RuleSnatEntry> getEntry(const uint32_t &id) = 0;
  virtual std::vector<std::shared_ptr<RuleSnatEntry>> getEntryList() = 0;
  virtual void addEntry(const uint32_t &id, const RuleSnatEntryJsonObject &conf) = 0;
  virtual void addEntryList(const std::vector<RuleSnatEntryJsonObject> &conf) = 0;
  virtual void replaceEntry(const uint32_t &id, const RuleSnatEntryJsonObject &conf) = 0;
  virtual void delEntry(const uint32_t &id) = 0;
  virtual void delEntryList() = 0;
  virtual RuleSnatAppendOutputJsonObject append(RuleSnatAppendInputJsonObject input) = 0;
};

