package net.floodlightcontroller.topology;
import net.floodlightcontroller.core.web.serializers.DPIDSerializer;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
public class NodePortTuple implements Comparable<NodePortTuple> {
    public NodePortTuple(DatapathId nodeId, OFPort portId) {
        this.nodeId = nodeId;
        this.portId = portId;
    }
    @JsonProperty("switch")
    @JsonSerialize(using=DPIDSerializer.class)
    public DatapathId getNodeId() {
        return nodeId;
    }
    public void setNodeId(DatapathId nodeId) {
        this.nodeId = nodeId;
    }
    @JsonProperty("port")
    public OFPort getPortId() {
        return portId;
    }
    public void setPortId(OFPort portId) {
        this.portId = portId;
    }
    public String toString() {
        return "[id=" + nodeId.toString() + ", port=" + portId.toString() + "]";
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        NodePortTuple other = (NodePortTuple) obj;
        if (!nodeId.equals(other.nodeId))
            return false;
        if (!portId.equals(other.portId))
            return false;
        return true;
    }
    public String toKeyString() {
        return (nodeId.toString()+ "|" + portId.toString());
    }
    @Override
    public int compareTo(NodePortTuple obj) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;
        if (this.getNodeId().getLong() < obj.getNodeId().getLong())
            return BEFORE;
        if (this.getNodeId().getLong() > obj.getNodeId().getLong())
            return AFTER;
        if (this.getPortId().getPortNumber() < obj.getPortId().getPortNumber())
            return BEFORE;
        if (this.getPortId().getPortNumber() > obj.getPortId().getPortNumber())
            return AFTER;
        return EQUAL;
    }
}
