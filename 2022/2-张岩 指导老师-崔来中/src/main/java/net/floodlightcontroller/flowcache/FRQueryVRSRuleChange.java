package net.floodlightcontroller.flowcache;
import java.util.Set;
@Deprecated
public class FRQueryVRSRuleChange extends FlowReconcileQuery {
    public Set<String> bvsNames;
    public FRQueryVRSRuleChange() {
        super(ReconcileQueryEvType.VRS_ROUTING_RULE_CHANGED);
    }
    public FRQueryVRSRuleChange(Set<String> bvsNames) {
        this();
        this.bvsNames = bvsNames;
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
        if (!(obj instanceof FRQueryVRSRuleChange)) {
            return false;
        }
        FRQueryVRSRuleChange other = (FRQueryVRSRuleChange) obj;
        if (! bvsNames.equals(other.bvsNames)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("BVS Names: ");
        builder.append(bvsNames);
        builder.append("]");
        return builder.toString();
    }
}
