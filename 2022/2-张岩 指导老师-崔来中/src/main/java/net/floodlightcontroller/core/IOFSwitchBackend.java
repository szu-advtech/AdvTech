package net.floodlightcontroller.core;
import java.util.Collection;
import java.util.List;
import org.projectfloodlight.openflow.protocol.OFBsnControllerConnectionsReply;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFPortDescStatsReply;
import org.projectfloodlight.openflow.protocol.OFPortStatus;
import org.projectfloodlight.openflow.protocol.OFTableFeaturesStatsReply;
import org.projectfloodlight.openflow.types.TableId;
import net.floodlightcontroller.util.OrderedCollection;
public interface IOFSwitchBackend extends IOFSwitch {
    void registerConnection(IOFConnectionBackend connection);
    void removeConnections();
    void removeConnection(IOFConnectionBackend connection);
    void setFeaturesReply(OFFeaturesReply featuresReply);
    OrderedCollection<PortChangeEvent> processOFPortStatus(OFPortStatus ps);
    void processOFTableFeatures(List<OFTableFeaturesStatsReply> replies);
    OrderedCollection<PortChangeEvent>
            comparePorts(Collection<OFPortDesc> ports);
    OrderedCollection<PortChangeEvent>
            setPorts(Collection<OFPortDesc> ports);
    void setSwitchProperties(SwitchDescription description);
    void setTableFull(boolean isFull);
    void startDriverHandshake();
    boolean isDriverHandshakeComplete();
    void processDriverHandshakeMessage(OFMessage m);
    void setPortDescStats(OFPortDescStatsReply portDescStats);
    void cancelAllPendingRequests();
    void setControllerRole(OFControllerRole role);
    void setStatus(SwitchStatus switchStatus);
    void updateControllerConnections(OFBsnControllerConnectionsReply controllerCxnsReply);
    boolean hasAnotherMaster();
    TableId getMaxTableForTableMissFlow();
    TableId setMaxTableForTableMissFlow(TableId max);
}
