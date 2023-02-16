package net.floodlightcontroller.flowcache;
import java.util.List;
import org.projectfloodlight.openflow.types.DatapathId;
@Deprecated
public class FRQueryBvsMatchSwitchPort extends FlowReconcileQuery {
    public DatapathId swId;
    public List<String> matchPortList;
    public FRQueryBvsMatchSwitchPort() {
        super(ReconcileQueryEvType.BVS_INTERFACE_RULE_CHANGED_MATCH_SWITCH_PORT);
    }
    public FRQueryBvsMatchSwitchPort(DatapathId swId, List<String> portList) {
        this();
        this.swId = swId;
        this.matchPortList = portList;
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
        if (!(obj instanceof FRQueryBvsMatchSwitchPort)) {
            return false;
        }
        FRQueryBvsMatchSwitchPort other = (FRQueryBvsMatchSwitchPort) obj;
        if (swId.equals(other.swId)) return false;
        if (!matchPortList.equals(other.matchPortList)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Switch: ");
        builder.append(swId.toString());
        builder.append(", Match Port List:");
        builder.append(matchPortList);
        builder.append("]");
        return builder.toString();
    }
}
