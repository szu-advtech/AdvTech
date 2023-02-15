package net.floodlightcontroller.core.module;
import java.util.Collection;
import java.util.Map;
public interface IFloodlightModuleContext {    
    public <T extends IFloodlightService> T getServiceImpl(Class<T> service);
    public Collection<Class<? extends IFloodlightService>> getAllServices();
    public Map<String, String> getConfigParams(IFloodlightModule module);
    public Map<String, String> getConfigParams(Class<? extends
                                                     IFloodlightModule> clazz);
}
