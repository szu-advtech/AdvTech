/*
 * k8switch API
 *
 * LoadBalancer Reverse-Proxy Service
 *
 * API version: 2.0.0
 * Generated by: Swagger Codegen (https://github.com/swagger-api/swagger-codegen.git)
 */

package swagger

type K8switch struct {

	// Name of the k8switch service
	Name string `json:"name,omitempty"`

	// UUID of the Cube
	Uuid string `json:"uuid,omitempty"`

	// Type of the Cube (TYPE_TC, TYPE_XDP_SKB, TYPE_XDP_DRV)
	Type_ string `json:"type,omitempty"`

	// Logging level of a cube, from none (OFF) to the most verbose (TRACE)
	Loglevel string `json:"loglevel,omitempty"`

	// Entry of the ports table
	Ports []Ports `json:"ports,omitempty"`

	// Range of VIPs where clusterIP services are exposed
	ClusterIpSubnet string `json:"cluster-ip-subnet,omitempty"`

	// Range of IPs of pods in this node
	ClientSubnet string `json:"client-subnet,omitempty"`

	// Range where client's IPs are mapped into
	VirtualClientSubnet string `json:"virtual-client-subnet,omitempty"`

	// Services (i.e., virtual ip:protocol:port) exported to the client
	Service []Service `json:"service,omitempty"`

	// Entry associated with the forwarding table
	FwdTable []FwdTable `json:"fwd-table,omitempty"`
}
