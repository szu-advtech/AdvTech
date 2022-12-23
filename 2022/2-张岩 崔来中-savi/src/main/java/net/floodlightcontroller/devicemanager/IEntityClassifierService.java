package net.floodlightcontroller.devicemanager;
import java.util.Collection;
import java.util.EnumSet;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import net.floodlightcontroller.devicemanager.internal.Entity;
public interface IEntityClassifierService extends IFloodlightService {
   IEntityClass classifyEntity(Entity entity);
   EnumSet<DeviceField> getKeyFields();
   IEntityClass reclassifyEntity(IDevice curDevice, Entity entity);
   void deviceUpdate(IDevice oldDevice, Collection<? extends IDevice> newDevices);
   public void addListener(IEntityClassListener listener);
}
