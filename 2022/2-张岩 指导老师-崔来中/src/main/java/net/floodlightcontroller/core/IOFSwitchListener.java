package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.types.DatapathId;
public interface IOFSwitchListener {
    public void switchAdded(DatapathId switchId);
    public void switchRemoved(DatapathId switchId);
    public void switchActivated(DatapathId switchId);
    public void switchPortChanged(DatapathId switchId,
                                  OFPortDesc port,
                                  PortChangeType type);
    public void switchChanged(DatapathId switchId);
}
