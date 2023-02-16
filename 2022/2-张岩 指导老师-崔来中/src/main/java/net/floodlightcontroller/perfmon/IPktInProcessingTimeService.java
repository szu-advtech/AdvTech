package net.floodlightcontroller.perfmon;
import java.util.List;
import org.projectfloodlight.openflow.protocol.OFMessage;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IPktInProcessingTimeService extends IFloodlightService {
    public void bootstrap(List<IOFMessageListener> listeners);
    public void recordStartTimeComp(IOFMessageListener listener);
    public void recordEndTimeComp(IOFMessageListener listener);
    public void recordStartTimePktIn();
    public void recordEndTimePktIn(IOFSwitch sw, OFMessage m, FloodlightContext cntx);
    public boolean isEnabled();
    public void setEnabled(boolean enabled);
    public CumulativeTimeBucket getCtb();
}
