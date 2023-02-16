package net.floodlightcontroller.flowcache;
import java.util.List;
@Deprecated
public class FRQueryBvsMatchTag extends FlowReconcileQuery {
    public List<String> tag;
    public FRQueryBvsMatchTag() {
        super(ReconcileQueryEvType.BVS_INTERFACE_RULE_CHANGED_MATCH_TAG);
    }
    public FRQueryBvsMatchTag(List<String> tag) {
        this();
        this.tag = tag;
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
        if (!(obj instanceof FRQueryBvsMatchTag)) {
            return false;
        }
        FRQueryBvsMatchTag other = (FRQueryBvsMatchTag) obj;
        if (! tag.equals(other.tag)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Tags: ");
        builder.append(tag);
        builder.append("]");
        return builder.toString();
    }
}
