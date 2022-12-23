package net.floodlightcontroller.perfmon;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.restserver.IRestApiService;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class PktInProcessingTime
    implements IFloodlightModule, IPktInProcessingTimeService {
	protected IFloodlightProviderService floodlightProvider;
    private IRestApiService restApi;
    protected long ptWarningThresholdInNano;
    protected static final String ControllerTableName = "controller_controller";
    public static final String COLUMN_ID = "id";
    public static final String COLUMN_PERF_MON = "performance_monitor_feature";
    protected static  Logger  logger = 
        LoggerFactory.getLogger(PktInProcessingTime.class);
    protected boolean isEnabled = false;
    protected boolean isInited = false;
    protected long lastPktTime_ns;
    private CumulativeTimeBucket ctb = null;
    protected static final long ONE_BUCKET_DURATION_NANOSECONDS  =
    @Override
    public void bootstrap(List<IOFMessageListener> listeners) {
            ctb = new CumulativeTimeBucket(listeners);
    }
    @Override
    public boolean isEnabled() {
        return isEnabled;
    }
    @Override
    public void setEnabled(boolean enabled) {
    	if(enabled){
    		bootstrap(floodlightProvider.getListeners().get(OFType.PACKET_IN));
    	}
        this.isEnabled = enabled;
        logger.debug("Setting module to " + isEnabled);
    }
    @Override
    public CumulativeTimeBucket getCtb() {
        return ctb;
    }
    private long startTimePktNs;
    private long startTimeCompNs;
    @Override
    public void recordStartTimeComp(IOFMessageListener listener) {
        if (isEnabled()) {
            startTimeCompNs = System.nanoTime();
        }
    }
    @Override
    public void recordEndTimeComp(IOFMessageListener listener) {
        if (isEnabled()) {
            long procTime = System.nanoTime() - startTimeCompNs;
            ctb.updateOneComponent(listener, procTime);
        }
    }
    @Override
    public void recordStartTimePktIn() {
        if (isEnabled()) {
            startTimePktNs = System.nanoTime();
        }
    }
    @Override
    public void recordEndTimePktIn(IOFSwitch sw, OFMessage m, FloodlightContext cntx) {
        if (isEnabled()) {
            long procTimeNs = System.nanoTime() - startTimePktNs;
            ctb.updatePerPacketCounters(procTimeNs);
            if (ptWarningThresholdInNano > 0 && 
                    procTimeNs > ptWarningThresholdInNano) {
                logger.warn("Time to process packet-in exceeded threshold: {}", 
                            procTimeNs/1000);
            }
        }
    }
    @Override
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
        Collection<Class<? extends IFloodlightService>> l = 
                new ArrayList<Class<? extends IFloodlightService>>();
        l.add(IRestApiService.class);
        l.add(IFloodlightProviderService.class);
        return l;
    }
    @Override
    public void init(FloodlightModuleContext context)
                                             throws FloodlightModuleException {
    	floodlightProvider = context
                .getServiceImpl(IFloodlightProviderService.class);
        restApi = context.getServiceImpl(IRestApiService.class);
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
        restApi.addRestletRoutable(new PerfWebRoutable());
        ptWarningThresholdInNano = Long.parseLong(System.getProperty(
        if (ptWarningThresholdInNano > 0) {
            logger.info("Packet processing time threshold for warning" +
            		" set to {} ms.", ptWarningThresholdInNano/1000000);
        }
    }
}
