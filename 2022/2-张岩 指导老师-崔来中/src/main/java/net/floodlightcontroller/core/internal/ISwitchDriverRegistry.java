package net.floodlightcontroller.core.internal;
import net.floodlightcontroller.core.IOFConnectionBackend;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.IOFSwitchDriver;
import net.floodlightcontroller.core.SwitchDescription;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.types.DatapathId;
public interface ISwitchDriverRegistry {
    void addSwitchDriver(String manufacturerDescriptionPrefix,
                         IOFSwitchDriver driver);
    IOFSwitchBackend getOFSwitchInstance(IOFConnectionBackend connection, SwitchDescription description, OFFactory factory, DatapathId datapathId);
}
