package net.floodlightcontroller.flowcache;
import java.util.List;
@Deprecated
public class FRQueryBvsMatchVlan extends FlowReconcileQuery {
    public List<Integer> vlans;
    public FRQueryBvsMatchVlan() {
        super(ReconcileQueryEvType.BVS_INTERFACE_RULE_CHANGED_MATCH_VLAN);
    }
    public FRQueryBvsMatchVlan(List<Integer> vlans) {
        this();
        this.vlans = vlans;
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
        if (!(obj instanceof FRQueryBvsMatchVlan)) {
            return false;
        }
        FRQueryBvsMatchVlan other = (FRQueryBvsMatchVlan) obj;
        if (! vlans.equals(other.vlans)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Vlans: ");
        builder.append(vlans);
        builder.append("]");
        return builder.toString();
    }
}
