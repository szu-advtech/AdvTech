/**
* k8sdispatcher API generated from k8sdispatcher.yang
*
* NOTE: This file is auto generated by polycube-codegen
* https://github.com/polycube-network/polycube-codegen
*/


/* Do not edit this file manually */

/*
* K8sdispatcherApi.h
*
*/

#pragma once

#define POLYCUBE_SERVICE_NAME "k8sdispatcher"


#include "polycube/services/response.h"
#include "polycube/services/shared_lib_elements.h"

#include "K8sdispatcherJsonObject.h"
#include "NodeportRuleJsonObject.h"
#include "PortsJsonObject.h"
#include "SessionRuleJsonObject.h"
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif

Response create_k8sdispatcher_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response create_k8sdispatcher_nodeport_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response create_k8sdispatcher_nodeport_rule_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response create_k8sdispatcher_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response create_k8sdispatcher_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response delete_k8sdispatcher_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response delete_k8sdispatcher_nodeport_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response delete_k8sdispatcher_nodeport_rule_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response delete_k8sdispatcher_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response delete_k8sdispatcher_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_internal_src_ip_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_nodeport_range_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_nodeport_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_nodeport_rule_external_traffic_policy_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_nodeport_rule_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_nodeport_rule_rule_name_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_ports_ip_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_ports_type_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_session_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_session_rule_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_session_rule_new_ip_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_session_rule_new_port_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_session_rule_operation_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sdispatcher_session_rule_originating_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response replace_k8sdispatcher_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response replace_k8sdispatcher_nodeport_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response replace_k8sdispatcher_nodeport_rule_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response replace_k8sdispatcher_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response replace_k8sdispatcher_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_nodeport_range_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_nodeport_rule_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_nodeport_rule_external_traffic_policy_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_nodeport_rule_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sdispatcher_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);

Response k8sdispatcher_list_by_id_help(const char *name, const Key *keys, size_t num_keys);
Response k8sdispatcher_nodeport_rule_list_by_id_help(const char *name, const Key *keys, size_t num_keys);
Response k8sdispatcher_ports_list_by_id_help(const char *name, const Key *keys, size_t num_keys);
Response k8sdispatcher_session_rule_list_by_id_help(const char *name, const Key *keys, size_t num_keys);


#ifdef __cplusplus
}
#endif

