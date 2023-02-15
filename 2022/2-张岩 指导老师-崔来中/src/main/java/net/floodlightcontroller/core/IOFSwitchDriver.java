package net.floodlightcontroller.core;
import net.floodlightcontroller.core.IOFSwitchBackend;
import org.projectfloodlight.openflow.protocol.OFFactory;
import net.floodlightcontroller.core.SwitchDescription;
public interface IOFSwitchDriver {
    public IOFSwitchBackend getOFSwitchImpl(SwitchDescription description, OFFactory factory);
}