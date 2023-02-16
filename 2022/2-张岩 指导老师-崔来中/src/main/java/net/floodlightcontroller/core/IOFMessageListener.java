package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFType;
public interface IOFMessageListener extends IListener<OFType> {
  public Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx);
}
