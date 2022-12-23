package net.floodlightcontroller.flowcache;
@Deprecated
public class FRQueryVRSArpChange extends FlowReconcileQuery {
    public String tenant;
    public FRQueryVRSArpChange() {
        super(ReconcileQueryEvType.VRS_STATIC_ARP_CHANGED);
    }
    public FRQueryVRSArpChange(String tenant) {
        this();
        this.tenant = tenant;
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
        if (!(obj instanceof FRQueryVRSArpChange)) {
            return false;
        }
        FRQueryVRSArpChange other = (FRQueryVRSArpChange) obj;
        if (! tenant.equals(other.tenant)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Tenant: ");
        builder.append(tenant);
        builder.append("]");
        return builder.toString();
    }
}
