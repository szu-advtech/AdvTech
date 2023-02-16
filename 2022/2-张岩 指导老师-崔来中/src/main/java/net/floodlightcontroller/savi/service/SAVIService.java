package net.floodlightcontroller.savi.service;
import java.util.List;
import org.projectfloodlight.openflow.protocol.match.Match;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.routing.IRoutingDecision.RoutingAction;
import net.floodlightcontroller.savi.action.Action;
public interface SAVIService {
	public boolean match(Ethernet eth);
	public List<Match> getMatches();
	public RoutingAction process(SwitchPort switchPort,Ethernet eth);
	public void pushActins(List<Action> actions);
	public void handlePortDown(SwitchPort switchPort);
}
