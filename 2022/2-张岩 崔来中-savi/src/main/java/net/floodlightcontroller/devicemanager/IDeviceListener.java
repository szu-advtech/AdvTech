package net.floodlightcontroller.devicemanager;
import net.floodlightcontroller.core.IListener;
public interface IDeviceListener extends IListener<String> {
    public void deviceAdded(IDevice device);
    public void deviceRemoved(IDevice device);
    public void deviceMoved(IDevice device);
    public void deviceIPV4AddrChanged(IDevice device);
    public void deviceIPV6AddrChanged(IDevice device);
    public void deviceVlanChanged(IDevice device);
}
