package net.floodlightcontroller.devicemanager;
import java.util.EnumSet;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import net.floodlightcontroller.devicemanager.internal.Device;
public interface IEntityClass {
    EnumSet<DeviceField> getKeyFields();
    String getName();
}
