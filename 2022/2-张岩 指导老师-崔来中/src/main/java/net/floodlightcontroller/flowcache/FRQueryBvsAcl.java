package net.floodlightcontroller.flowcache;
@Deprecated
public class FRQueryBvsAcl extends FlowReconcileQuery {
    public String bvsName;
    public String bvsInterfaceName;
    public DIRECTION direction;
    public enum DIRECTION {
        INGRESS,
        EGRESS,
    };
    public FRQueryBvsAcl() {
        super(ReconcileQueryEvType.ACL_CONFIG_CHANGED);
    }
    public FRQueryBvsAcl(String bvsName, String bvsInterfaceName, DIRECTION direction) {
        this();
        this.bvsName = bvsName;
        this.bvsInterfaceName = bvsInterfaceName;
        this.direction = direction;
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
        if (!(obj instanceof FRQueryBvsAcl)) {
            return false;
        }
        FRQueryBvsAcl other = (FRQueryBvsAcl) obj;
        if (! bvsName.equals(other.bvsName)) return false;
        if (! bvsInterfaceName.equals(other.bvsInterfaceName)) return false;
        if (! direction.equals(other.direction)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("BVS Name: ");
        builder.append(bvsName);
        builder.append(", BVS Interface Name: ");
        builder.append(bvsInterfaceName);
        builder.append(", ACL Direction: ");
        builder.append(direction);
        builder.append("]");
        return builder.toString();
    }
}
