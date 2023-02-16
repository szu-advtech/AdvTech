package net.floodlightcontroller.flowcache;
@Deprecated
public class FRQueryBvsMatchSubnet extends FlowReconcileQuery {
    public String ipSubnet;
    public FRQueryBvsMatchSubnet() {
        super(ReconcileQueryEvType.BVS_INTERFACE_RULE_CHANGED_MATCH_IPSUBNET);
    }
    public FRQueryBvsMatchSubnet(String ipSubnet) {
        this();
        this.ipSubnet = ipSubnet;
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
        if (!(obj instanceof FRQueryBvsMatchSubnet)) {
            return false;
        }
        FRQueryBvsMatchSubnet other = (FRQueryBvsMatchSubnet) obj;
        if (! ipSubnet.equals(other.ipSubnet)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("IP Subnet: ");
        builder.append(ipSubnet);
        builder.append("]");
        return builder.toString();
    }
}
