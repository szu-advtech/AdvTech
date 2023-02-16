package net.floodlightcontroller.flowcache;
import org.projectfloodlight.openflow.types.MacAddress;
@Deprecated
public class FRQueryBvsMatchMac extends FlowReconcileQuery {
    public String mac;
    public FRQueryBvsMatchMac() {
        super(ReconcileQueryEvType.BVS_INTERFACE_RULE_CHANGED_MATCH_MAC);
    }
    public FRQueryBvsMatchMac(String mac) {
        this();
        this.mac = mac;
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
        if (!(obj instanceof FRQueryBvsMatchMac)) {
            return false;
        }
        FRQueryBvsMatchMac other = (FRQueryBvsMatchMac) obj;
        if (! mac.equals(other.mac)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("MAC: ");
        builder.append(MacAddress.of(mac).toString());
        builder.append("]");
        return builder.toString();
    }
}
