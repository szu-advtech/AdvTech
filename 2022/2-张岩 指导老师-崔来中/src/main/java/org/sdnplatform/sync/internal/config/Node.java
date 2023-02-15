package org.sdnplatform.sync.internal.config;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
public class Node {
    private String hostname;
    private int port;
    private short nodeId;
    private short domainId;
    @JsonCreator
    public Node(@JsonProperty("hostname") String hostname, 
                @JsonProperty("port") int port,
                @JsonProperty("nodeId") short nodeId,
                @JsonProperty("domainId") short domainId) {
        super();
        this.hostname = hostname;
        this.port = port;
        this.nodeId = nodeId;
        this.domainId = domainId;
    }
    public String getHostname() {
        return hostname;
    }
    public int getPort() {
        return port;
    }
    public short getNodeId() {
        return nodeId;
    }
    public short getDomainId() {
        return domainId;
    }
    @Override
    public String toString() {
        return "Node [hostname=" + hostname + ", port=" + port + ", nodeId="
                + nodeId + ", domainId=" + domainId + "]";
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result =
                        + ((hostname == null) ? 0 : hostname.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        Node other = (Node) obj;
        if (domainId != other.domainId) return false;
        if (hostname == null) {
            if (other.hostname != null) return false;
        } else if (!hostname.equals(other.hostname)) return false;
        if (nodeId != other.nodeId) return false;
        if (port != other.port) return false;
        return true;
    }
}
