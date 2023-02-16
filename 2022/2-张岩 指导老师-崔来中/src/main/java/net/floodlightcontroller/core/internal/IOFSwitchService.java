package net.floodlightcontroller.core.internal;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitchDriver;
import net.floodlightcontroller.core.IOFSwitchListener;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.rest.SwitchRepresentation;
import org.projectfloodlight.openflow.types.DatapathId;
public interface IOFSwitchService extends IFloodlightService {
    Map<DatapathId, IOFSwitch> getAllSwitchMap();
    IOFSwitch getSwitch(DatapathId dpid);
    IOFSwitch getActiveSwitch(DatapathId dpid);
    void addOFSwitchListener(IOFSwitchListener listener);
    void addOFSwitchDriver(String manufacturerDescriptionPrefix, IOFSwitchDriver driver);
    void removeOFSwitchListener(IOFSwitchListener listener);
    void registerLogicalOFMessageCategory(LogicalOFMessageCategory category);
    void registerHandshakePlugin(IAppHandshakePluginFactory plugin);
    List<SwitchRepresentation> getSwitchRepresentations();
    SwitchRepresentation getSwitchRepresentation(DatapathId dpid);
    Set<DatapathId> getAllSwitchDpids();
    List<OFSwitchHandshakeHandler> getSwitchHandshakeHandlers();
}
