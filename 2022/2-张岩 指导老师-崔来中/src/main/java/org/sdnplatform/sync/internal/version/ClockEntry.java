package org.sdnplatform.sync.internal.version;
import java.io.Serializable;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
public final class ClockEntry
    implements Cloneable, Serializable {
    private static final long serialVersionUID = -759862327985468981L;
    private final short nodeId;
    private final long version;
    @JsonCreator
    public ClockEntry(@JsonProperty("nodeId") short nodeId, 
                      @JsonProperty("version") long version) {
        if(nodeId < 0)
            throw new IllegalArgumentException("Node id " + nodeId + " is not in the range (0, "
                                               + Short.MAX_VALUE + ").");
        if(version < 1)
            throw new IllegalArgumentException("Version " + version + " is not in the range (1, "
                                               + Short.MAX_VALUE + ").");
        this.nodeId = nodeId;
        this.version = version;
    }
    @Override
    public ClockEntry clone() {
        try {
            return (ClockEntry) super.clone();
        } catch(CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
    public short getNodeId() {
        return nodeId;
    }
    public long getVersion() {
        return version;
    }
    public ClockEntry incremented() {
        return new ClockEntry(nodeId, version + 1);
    }
    @Override
    public int hashCode() {
        return nodeId + (((int) version) << 16);
    }
    @Override
    public boolean equals(Object o) {
        if(this == o)
            return true;
        if(o == null)
            return false;
        if(o.getClass().equals(ClockEntry.class)) {
            ClockEntry v = (ClockEntry) o;
            return v.getNodeId() == getNodeId() && v.getVersion() == getVersion();
        } else {
            return false;
        }
    }
    @Override
    public String toString() {
        return nodeId + ":" + version;
    }
}
