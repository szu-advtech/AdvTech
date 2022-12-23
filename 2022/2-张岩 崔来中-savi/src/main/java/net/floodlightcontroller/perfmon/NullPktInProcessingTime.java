package net.floodlightcontroller.perfmon;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.protocol.OFMessage;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
public class NullPktInProcessingTime 
    implements IFloodlightModule, IPktInProcessingTimeService {
    private CumulativeTimeBucket ctb;
    private boolean inited = false;
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        Collection<Class<? extends IFloodlightService>> l = 
                new ArrayList<Class<? extends IFloodlightService>>();
        l.add(IPktInProcessingTimeService.class);
        return l;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
            getServiceImpls() {
        Map<Class<? extends IFloodlightService>,
        IFloodlightService> m = 
            new HashMap<Class<? extends IFloodlightService>,
                        IFloodlightService>();
        m.put(IPktInProcessingTimeService.class, this);
        return m;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
        return null;
    }
    @Override
    public void init(FloodlightModuleContext context)
                             throws FloodlightModuleException {
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
    }
    @Override
    public boolean isEnabled() {
        return false;
    }
    @Override
    public void bootstrap(List<IOFMessageListener> listeners) {
        if (!inited)
            ctb = new CumulativeTimeBucket(listeners);
    }
    @Override
    public void recordStartTimeComp(IOFMessageListener listener) {
    }
    @Override
    public void recordEndTimeComp(IOFMessageListener listener) {
    }
    @Override
    public void recordStartTimePktIn() {
    }
    @Override
    public void recordEndTimePktIn(IOFSwitch sw, OFMessage m,
                                   FloodlightContext cntx) {
    }
    @Override
    public void setEnabled(boolean enabled) {
    }
    @Override
    public CumulativeTimeBucket getCtb() {
        return ctb;
    }
}
