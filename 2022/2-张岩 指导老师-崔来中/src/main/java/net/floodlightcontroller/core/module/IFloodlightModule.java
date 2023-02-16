package net.floodlightcontroller.core.module;
import java.util.Collection;
import java.util.Map;
public interface IFloodlightModule {
    public Collection<Class<? extends IFloodlightService>> getModuleServices();
    public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls();
    public Collection<Class<? extends IFloodlightService>> getModuleDependencies();
    void init(FloodlightModuleContext context) throws FloodlightModuleException;
    void startUp(FloodlightModuleContext context) throws FloodlightModuleException;
}
