package net.floodlightcontroller.flowcache;
import net.floodlightcontroller.core.FloodlightContextStore;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightService;
@Deprecated
public interface IFlowReconcileEngineService extends IFloodlightService {
    public static final FloodlightContextStore<String> fcStore =
        new FloodlightContextStore<String>();
    public static final String FLOWRECONCILE_APP_INSTANCE_NAME = "net.floodlightcontroller.flowcache.appInstanceName";
    public void submitFlowQueryEvent(FlowReconcileQuery query);
    public void updateFlush();
    public void init(FloodlightModuleContext fmc) throws FloodlightModuleException;
    public void startUp(FloodlightModuleContext fmc);
}
