package net.floodlightcontroller.devicemanager.internal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.IEntityClassListener;
import net.floodlightcontroller.devicemanager.IEntityClassifierService;
public class DefaultEntityClassifier implements
        IEntityClassifierService,
        IFloodlightModule 
{
    protected static class DefaultEntityClass implements IEntityClass {
        String name;
        public DefaultEntityClass(String name) {
            this.name = name;
        }
        @Override
        public EnumSet<IDeviceService.DeviceField> getKeyFields() {
            return keyFields;
        }
        @Override
        public String getName() {
            return name;
        }
    }
    protected static EnumSet<DeviceField> keyFields;
    static {
        keyFields = EnumSet.of(DeviceField.MAC, DeviceField.VLAN);
    }
    protected static DefaultEntityClass entityClass = new DefaultEntityClass("DefaultEntityClass");
    @Override
    public IEntityClass classifyEntity(Entity entity) {
        return entityClass;
    }
    @Override
    public IEntityClass reclassifyEntity(IDevice curDevice, Entity entity) {
        return entityClass;
    }
    @Override
    public void deviceUpdate(IDevice oldDevice, Collection<? extends IDevice> newDevices) {
    }
    @Override
    public EnumSet<DeviceField> getKeyFields() {
        return keyFields;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        Collection<Class<? extends IFloodlightService>> l =  new ArrayList<Class<? extends IFloodlightService>>();
        l.add(IEntityClassifierService.class);
        return l;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
        Map<Class<? extends IFloodlightService>,
        IFloodlightService> m = new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
        m.put(IEntityClassifierService.class, this);
        return m;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
        return null;
    }
    @Override
    public void init(FloodlightModuleContext context) throws FloodlightModuleException {
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
    }
    @Override
    public void addListener(IEntityClassListener listener) {
    }
}