package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.protocol.OFMessage;
public interface IControllerCompletionListener {
	public void onMessageConsumed(IOFSwitch sw, OFMessage msg, FloodlightContext cntx);
	public String getName();
}
