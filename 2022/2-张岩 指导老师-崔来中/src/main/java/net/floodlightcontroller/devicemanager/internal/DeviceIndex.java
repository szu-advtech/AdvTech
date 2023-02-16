package net.floodlightcontroller.devicemanager.internal;
import java.util.Collection;
import java.util.EnumSet;
import java.util.Iterator;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
public abstract class DeviceIndex {
    protected EnumSet<DeviceField> keyFields;
    public DeviceIndex(EnumSet<DeviceField> keyFields) {
        super();
        this.keyFields = keyFields;
    }
    public abstract Iterator<Long> queryByEntity(Entity entity);
    public abstract Iterator<Long> getAll();
    public abstract boolean updateIndex(Device device, Long deviceKey);
    public abstract boolean updateIndex(Entity entity, Long deviceKey);
    public abstract void removeEntity(Entity entity);
    public abstract void removeEntity(Entity entity, Long deviceKey);
    public void removeEntityIfNeeded(Entity entity, Long deviceKey,
                                     Collection<Entity> others) {
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        for (Entity o : others) {
            IndexedEntity oio = new IndexedEntity(keyFields, o);
            if (oio.equals(ie)) return;
        }
        Iterator<Long> keyiter = this.queryByEntity(entity);
        while (keyiter.hasNext()) {
                Long key = keyiter.next();
                if (key.equals(deviceKey)) {
                    removeEntity(entity, deviceKey);
                    break;
                }
        }
    }
}