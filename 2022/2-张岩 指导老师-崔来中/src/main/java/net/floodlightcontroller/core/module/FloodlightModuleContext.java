package net.floodlightcontroller.core.module;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
public class FloodlightModuleContext implements IFloodlightModuleContext {
    protected Map<Class<? extends IFloodlightService>, IFloodlightService> serviceMap;
    protected Map<Class<? extends IFloodlightModule>, Map<String, String>> configParams;
    private final FloodlightModuleLoader moduleLoader;
    public FloodlightModuleContext(FloodlightModuleLoader moduleLoader) {
        serviceMap = 
                new HashMap<Class<? extends IFloodlightService>,
                                      IFloodlightService>();
        configParams =
                new HashMap<Class<? extends IFloodlightModule>,
                                Map<String, String>>();
        this.moduleLoader = moduleLoader;
    }
    public FloodlightModuleContext() {
        this(null);
    }
    @SuppressWarnings("unchecked")
    @Override
    public <T extends IFloodlightService> T getServiceImpl(Class<T> service) {
        IFloodlightService s = serviceMap.get(service);
        return (T)s;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getAllServices() {
        return serviceMap.keySet();
    }
    @Override
    public Map<String, String> getConfigParams(IFloodlightModule module) {
        Class<? extends IFloodlightModule> clazz = module.getClass();
        return getConfigParams(clazz);
    }
    @Override
    public Map<String, String> getConfigParams(Class<? extends IFloodlightModule> clazz) {
        Map<String, String> retMap = configParams.get(clazz);
        if (retMap == null) {
            retMap = new HashMap<String, String>();
            configParams.put(clazz, retMap);
        }
        for (Class<? extends IFloodlightModule> c : configParams.keySet()) {
            if (c.isAssignableFrom(clazz)) {
                for (Map.Entry<String, String> ent : configParams.get(c).entrySet()) {
                    if (!retMap.containsKey(ent.getKey())) {
                        retMap.put(ent.getKey(), ent.getValue());
                    }
                }
            }
        }
        return retMap;
    }
    public void addConfigParam(IFloodlightModule mod, String key, String value) {
        Map<String, String> moduleParams = configParams.get(mod.getClass());
        if (moduleParams == null) {
            moduleParams = new HashMap<String, String>();
            configParams.put(mod.getClass(), moduleParams);
        }
        moduleParams.put(key, value);
    }
    public FloodlightModuleLoader getModuleLoader() {
        return moduleLoader;
    }
    public void addService(Class<? extends IFloodlightService> clazz, 
                           IFloodlightService service) {
        serviceMap.put(clazz, service);
    }
 }
