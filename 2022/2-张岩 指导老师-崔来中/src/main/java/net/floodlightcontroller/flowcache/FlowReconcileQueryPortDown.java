package net.floodlightcontroller.flowcache;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
@Deprecated
public class FlowReconcileQueryPortDown extends FlowReconcileQuery {
    public DatapathId swId;
    public OFPort port;
    public FlowReconcileQueryPortDown() {
        super(ReconcileQueryEvType.LINK_DOWN);
    }
    public FlowReconcileQueryPortDown(DatapathId swId, OFPort portDown) {
        this();
        this.swId = swId;
        this.port = portDown;
    }
    @Override
    public int hashCode() {
        final int prime = 347;
        int result = super.hashCode();
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!super.equals(obj)) {
            return false;
        }
        if (!(obj instanceof FlowReconcileQueryPortDown)) {
            return false;
        }
        FlowReconcileQueryPortDown other = (FlowReconcileQueryPortDown) obj;
        if (swId != other.swId) return false;
        if (port != other.port) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Switch: ");
        builder.append(swId.toString());
        builder.append(", Port: ");
        builder.append(port.getPortNumber());
        builder.append("]");
        return builder.toString();
    }
}
