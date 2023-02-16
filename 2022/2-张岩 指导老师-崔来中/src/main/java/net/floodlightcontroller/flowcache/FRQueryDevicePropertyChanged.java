package net.floodlightcontroller.flowcache;
import net.floodlightcontroller.devicemanager.IDevice;
@Deprecated
public class FRQueryDevicePropertyChanged extends FlowReconcileQuery {
    public IDevice device;
    public FRQueryDevicePropertyChanged() {
        super(ReconcileQueryEvType.DEVICE_PROPERTY_CHANGED);
    }
    public FRQueryDevicePropertyChanged(IDevice device) {
        this();
        this.device = device;
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
        if (!(obj instanceof FRQueryDevicePropertyChanged)) {
            return false;
        }
        FRQueryDevicePropertyChanged other = (FRQueryDevicePropertyChanged) obj;
        if (! device.equals(other.device)) return false;
        return true;
    }
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        builder.append("Device: ");
        builder.append(device.getMACAddress().toString());
        builder.append("]");
        return builder.toString();
    }
}
