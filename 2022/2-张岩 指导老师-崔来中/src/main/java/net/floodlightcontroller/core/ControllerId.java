package net.floodlightcontroller.core;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.internal.config.ClusterConfig;
import com.google.common.base.Optional;
public class ControllerId {
    private final short nodeId;
    private ControllerId(short nodeId) {
        if(nodeId == ClusterConfig.NODE_ID_UNCONFIGURED)
            throw new IllegalArgumentException("nodeId is unconfigured");
        this.nodeId = nodeId;
    }
    public short getNodeId() {
        return nodeId;
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
        ControllerId other = (ControllerId) obj;
        if (nodeId != other.nodeId)
            return false;
        return true;
    }
    @Override
    public String toString() {
        return Short.toString(nodeId);
    }
    public static ControllerId of(short nodeId) {
        return new ControllerId(nodeId);
    }
}
