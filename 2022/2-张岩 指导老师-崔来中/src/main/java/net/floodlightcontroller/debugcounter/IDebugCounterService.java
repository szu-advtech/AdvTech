package net.floodlightcontroller.debugcounter;
import java.util.List;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IDebugCounterService extends IFloodlightService {
    public enum MetaData {
        WARN,
        DROP,
        ERROR
    }
    public boolean registerModule(String moduleName);
    public IDebugCounter
    registerCounter(String moduleName, String counterHierarchy,
                    String counterDescription, MetaData... metaData);
    public boolean resetCounterHierarchy(String moduleName, String counterHierarchy);
    public void resetAllCounters();
    public boolean resetAllModuleCounters(String moduleName);
    public boolean removeCounterHierarchy(String moduleName, String counterHierarchy);
    public List<DebugCounterResource>
    getCounterHierarchy(String moduleName, String counterHierarchy);
    public List<DebugCounterResource> getAllCounterValues();
    public  List<DebugCounterResource> getModuleCounterValues(String moduleName);
}
