package net.floodlightcontroller.devicemanager.internal;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
public class DeviceUniqueIndex extends DeviceIndex {
    private final ConcurrentHashMap<IndexedEntity, Long> index;
    public DeviceUniqueIndex(EnumSet<DeviceField> keyFields) {
        super(keyFields);
        index = new ConcurrentHashMap<IndexedEntity, Long>();
    }
    @Override
    public Iterator<Long> queryByEntity(Entity entity) {
        final Long deviceKey = findByEntity(entity);
        if (deviceKey != null)
            return Collections.<Long>singleton(deviceKey).iterator();
        return Collections.<Long>emptySet().iterator();
    }
    @Override
    public Iterator<Long> getAll() {
        return index.values().iterator();
    }
    @Override
    public boolean updateIndex(Device device, Long deviceKey) {
        for (Entity e : device.entities) {
            IndexedEntity ie = new IndexedEntity(keyFields, e);
            if (!ie.hasNonZeroOrNonNullKeys()) continue;
            Long ret = index.putIfAbsent(ie, deviceKey);
            if (ret != null && !ret.equals(deviceKey)) {
                return false;
            }
        }
        return true;
    }
    @Override
    public boolean updateIndex(Entity entity, Long deviceKey) {
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        if (!ie.hasNonZeroOrNonNullKeys()) return false;
        index.put(ie, deviceKey);
        return true;
    }
    @Override
    public void removeEntity(Entity entity) {
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        index.remove(ie);
    }
    @Override
    public void removeEntity(Entity entity, Long deviceKey) {
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        index.remove(ie, deviceKey);
    }
    public Long findByEntity(Entity entity) {
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        Long deviceKey = index.get(ie);
        if (deviceKey == null)
            return null;
        return deviceKey;
    }
}