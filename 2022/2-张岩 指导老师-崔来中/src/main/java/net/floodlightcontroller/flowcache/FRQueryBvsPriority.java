package net.floodlightcontroller.flowcache;
@Deprecated
public class FRQueryBvsPriority extends FlowReconcileQuery {
    public FRQueryBvsPriority() {
        super(ReconcileQueryEvType.BVS_PRIORITY_CHANGED);
    }
    public FRQueryBvsPriority(int lowP, int highP) {
        this();
        this.lowP = lowP;
        this.highP = highP;
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
        if (!(obj instanceof FRQueryBvsPriority)) {
            return false;
        }
        FRQueryBvsPriority other = (FRQueryBvsPriority) obj;
        if (lowP != other.lowP) return false;
        if (highP != other.highP) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Lower Priority: ");
        builder.append(lowP);
        builder.append("Higher Priority: ");
        builder.append(highP);
        builder.append("]");
        return builder.toString();
    }
}
