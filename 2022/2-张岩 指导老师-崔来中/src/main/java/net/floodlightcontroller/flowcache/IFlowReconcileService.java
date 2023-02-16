package net.floodlightcontroller.flowcache;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.flowcache.PriorityPendingQueue.EventPriority;
@Deprecated
public interface IFlowReconcileService extends IFloodlightService {
    public void addFlowReconcileListener(IFlowReconcileListener listener);
    public void removeFlowReconcileListener(IFlowReconcileListener listener);
    public void clearFlowReconcileListeners();
    public void reconcileFlow(OFMatchReconcile ofmRcIn, EventPriority priority) ;
    public void init(FloodlightModuleContext context)  throws FloodlightModuleException ;
    public void startUp(FloodlightModuleContext context)  throws FloodlightModuleException ;
}
