package net.floodlightcontroller.devicemanager.internal;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import net.floodlightcontroller.util.IterableIterator;
public class DeviceMultiIndex extends DeviceIndex {
    private ConcurrentHashMap<IndexedEntity, Collection<Long>> index;
    public DeviceMultiIndex(EnumSet<DeviceField> keyFields) {
        super(keyFields);
        index = new ConcurrentHashMap<IndexedEntity, Collection<Long>>();
    }
    @Override
    public Iterator<Long> queryByEntity(Entity entity) {
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        Collection<Long> devices = index.get(ie);
        if (devices != null)
            return devices.iterator();
        return Collections.<Long>emptySet().iterator();
    }
    @Override
    public Iterator<Long> getAll() {
        Iterator<Collection<Long>> iter = index.values().iterator();
        return new IterableIterator<Long>(iter);
    }
    @Override
    public boolean updateIndex(Device device, Long deviceKey) {
        for (Entity e : device.entities) {
            updateIndex(e, deviceKey);
        }
        return true;
    }
    @Override
    public boolean updateIndex(Entity entity, Long deviceKey) {
        Collection<Long> devices = null;
        IndexedEntity ie = new IndexedEntity(keyFields, entity);
        if (!ie.hasNonZeroOrNonNullKeys()) return false;
        devices = index.get(ie);
        if (devices == null) {
            Map<Long,Boolean> chm = new ConcurrentHashMap<Long,Boolean>();
            devices = Collections.newSetFromMap(chm);
            Collection<Long> r = index.putIfAbsent(ie, devices);
            if (r != null)
                devices = r;
        }
        devices.add(deviceKey);
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
        Collection<Long> devices = index.get(ie);
        if (devices != null)
            devices.remove(deviceKey);
    }
}