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

/*
* K8sfilterApi.h
*
*/

#pragma once

#define POLYCUBE_SERVICE_NAME "k8sfilter"


#include "polycube/services/response.h"
#include "polycube/services/shared_lib_elements.h"

#include "K8sfilterJsonObject.h"
#include "PortsJsonObject.h"
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif

Response create_k8sfilter_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response create_k8sfilter_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response create_k8sfilter_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response delete_k8sfilter_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response delete_k8sfilter_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response delete_k8sfilter_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sfilter_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sfilter_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sfilter_nodeport_range_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sfilter_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sfilter_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response read_k8sfilter_ports_type_by_id_handler(const char *name, const Key *keys, size_t num_keys);
Response replace_k8sfilter_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response replace_k8sfilter_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response replace_k8sfilter_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sfilter_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sfilter_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sfilter_nodeport_range_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sfilter_ports_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);
Response update_k8sfilter_ports_list_by_id_handler(const char *name, const Key *keys, size_t num_keys, const char *value);

Response k8sfilter_list_by_id_help(const char *name, const Key *keys, size_t num_keys);
Response k8sfilter_ports_list_by_id_help(const char *name, const Key *keys, size_t num_keys);


#ifdef __cplusplus
}
#endif

