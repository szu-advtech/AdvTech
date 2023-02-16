package net.floodlightcontroller.routing;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.SwitchPort;
public class RoutingDecision implements IRoutingDecision {
    protected RoutingAction action;
    protected Match match;
    protected int hardTimeout;
    protected SwitchPort srcPort;
    protected IDevice srcDevice;
    protected List<IDevice> destDevices;
    protected List<SwitchPort> broadcastIntertfaces;
    public RoutingDecision(DatapathId swDipd,
                                  OFPort inPort,
                                  IDevice srcDevice,
                                  RoutingAction action) {
        this.srcPort = new SwitchPort(swDipd, inPort);
        this.srcDevice = srcDevice;
        this.destDevices = Collections.synchronizedList(new ArrayList<IDevice>());
        this.broadcastIntertfaces = Collections.synchronizedList(new ArrayList<SwitchPort>());
        this.action = action;
        this.match = null;
        this.hardTimeout = ForwardingBase.FLOWMOD_DEFAULT_HARD_TIMEOUT;
    }
    @Override
    public RoutingAction getRoutingAction() {
        return this.action;
    }
    @Override
    public void setRoutingAction(RoutingAction action) {
        this.action = action;
    }
    @Override
    public SwitchPort getSourcePort() {
        return this.srcPort;
    }
    @Override
    public IDevice getSourceDevice() {
        return this.srcDevice;
    }
    @Override
    public List<IDevice> getDestinationDevices() {
        return this.destDevices;
    }
    @Override
    public void addDestinationDevice(IDevice d) {
        if (!destDevices.contains(d)) {
            destDevices.add(d);
        }
    }
    @Override
    public void setMulticastInterfaces(List<SwitchPort> lspt) {
        this.broadcastIntertfaces = lspt;
    }
    @Override
    public List<SwitchPort> getMulticastInterfaces() {
        return this.broadcastIntertfaces;
    }
    @Override
    public Match getMatch() {
        return this.match;
    }
    @Override
    public void setMatch(Match match) {
        this.match = match;
    }
    @Override
    public int getHardTimeout() {
        return hardTimeout;
    }
    @Override
    public void setHardTimeout(short hardTimeout) {
        this.hardTimeout = hardTimeout;
    }
    @Override
    public void addToContext(FloodlightContext cntx) {
        rtStore.put(cntx, IRoutingDecision.CONTEXT_DECISION, this);
    }
    public String toString() {
        return "action " + action +
               " wildcard " +
               ((match == null) ? null : match.toString());
    }
}
