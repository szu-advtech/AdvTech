package net.floodlightcontroller.flowcache;
import java.util.Arrays;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.SwitchPort;
@Deprecated
public class FlowReconcileQueryDeviceMove extends FlowReconcileQuery {
    public IDevice deviceMoved;
    public SwitchPort[] oldAp;
    public FlowReconcileQueryDeviceMove() {
        super(ReconcileQueryEvType.DEVICE_MOVED);
    }
    public FlowReconcileQueryDeviceMove(IDevice deviceMoved, SwitchPort[] oldAp) {
        this();
        this.deviceMoved = deviceMoved;
        this.oldAp = oldAp.clone();
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
        if (!(obj instanceof FlowReconcileQueryDeviceMove)) {
            return false;
        }
        FlowReconcileQueryDeviceMove other = (FlowReconcileQueryDeviceMove) obj;
        if (oldAp == null) {
            if (other.oldAp != null) return false;
        } else if (!Arrays.equals(oldAp, other.oldAp)) return false;
        if (deviceMoved == null) {
            if (other.deviceMoved != null) return false;
        } else if (!deviceMoved.equals(other.deviceMoved)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Device: ");
        builder.append(deviceMoved.getMACAddress().toString());
        builder.append(", Old Attachment Points:");
        builder.append(Arrays.toString(oldAp));
        builder.append("]");
        return builder.toString();
    }
}
