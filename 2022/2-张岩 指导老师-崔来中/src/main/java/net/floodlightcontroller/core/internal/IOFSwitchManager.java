package net.floodlightcontroller.core.internal;
import java.util.List;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IOFConnectionBackend;
import net.floodlightcontroller.core.IOFSwitch.SwitchStatus;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.IOFSwitchDriver;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.SwitchDescription;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.types.DatapathId;
import com.google.common.collect.ImmutableList;
public interface IOFSwitchManager {
    void switchAdded(IOFSwitchBackend sw);
    void switchDisconnected(IOFSwitchBackend sw);
    void notifyPortChanged(IOFSwitchBackend sw, OFPortDesc port,
                           PortChangeType type);
    IOFSwitchBackend getOFSwitchInstance(IOFConnectionBackend connection,
                                         SwitchDescription description,
                                         OFFactory factory,
                                         DatapathId datapathId);
    void handleMessage(IOFSwitchBackend sw, OFMessage m, FloodlightContext bContext);
    public void handleOutgoingMessage(IOFSwitch sw, OFMessage m);
    ImmutableList<OFSwitchHandshakeHandler> getSwitchHandshakeHandlers();
    void addOFSwitchDriver(String manufacturerDescriptionPrefix,
                           IOFSwitchDriver driver);
    void switchStatusChanged(IOFSwitchBackend sw, SwitchStatus oldStatus,
            SwitchStatus newStatus);
    int getNumRequiredConnections();
    public void addSwitchEvent(DatapathId switchDpid, String reason, boolean flushNow);
    List<IAppHandshakePluginFactory> getHandshakePlugins();
    SwitchManagerCounters getCounters();
    boolean isCategoryRegistered(LogicalOFMessageCategory category);
    void handshakeDisconnected(DatapathId dpid);
}
