package net.floodlightcontroller.routing;
import java.util.List;
import org.projectfloodlight.openflow.protocol.match.Match;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.FloodlightContextStore;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.SwitchPort;
public interface IRoutingDecision {
    public enum RoutingAction {
        NONE, DROP, FORWARD, FORWARD_OR_FLOOD, MULTICAST
    }
    public static final FloodlightContextStore<IRoutingDecision> rtStore =
        new FloodlightContextStore<IRoutingDecision>();
    public static final String CONTEXT_DECISION =
            "net.floodlightcontroller.routing.decision";
    public void addToContext(FloodlightContext cntx);
    public RoutingAction getRoutingAction();
    public void setRoutingAction(RoutingAction action);
    public SwitchPort getSourcePort();
    public IDevice getSourceDevice();
    public List<IDevice> getDestinationDevices();
    public void addDestinationDevice(IDevice d);
    public List<SwitchPort> getMulticastInterfaces();
    public void setMulticastInterfaces(List<SwitchPort> lspt);
    public Match getMatch();
    public void setMatch(Match match);
    public int getHardTimeout();
    public void setHardTimeout(short hardTimeout);
}
