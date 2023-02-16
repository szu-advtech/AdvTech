package net.floodlightcontroller.firewall;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.match.Match;
public class AllowDropPair {
    public Match.Builder allow;
    public Match.Builder drop;
    @SuppressWarnings("unused")
	private AllowDropPair() {};
    public AllowDropPair(OFFactory factory) {
    	allow = factory.buildMatch();
    	drop = factory.buildMatch();
    }
}
