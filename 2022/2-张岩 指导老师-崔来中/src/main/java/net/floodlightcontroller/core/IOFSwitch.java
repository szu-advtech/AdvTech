package net.floodlightcontroller.core;
import java.net.SocketAddress;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Date;
import org.projectfloodlight.openflow.protocol.OFActionType;
import org.projectfloodlight.openflow.protocol.OFCapabilities;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFRequest;
import org.projectfloodlight.openflow.protocol.OFStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsRequest;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U64;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import net.floodlightcontroller.core.internal.OFConnection;
import net.floodlightcontroller.core.internal.TableFeatures;
import net.floodlightcontroller.core.web.serializers.IOFSwitchSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
@JsonSerialize(using=IOFSwitchSerializer.class)
public interface IOFSwitch extends IOFMessageWriter {
    public static final String SWITCH_DESCRIPTION_FUTURE = "description-future";
    public static final String SWITCH_DESCRIPTION_DATA = "description-data";
    public static final String SWITCH_SUPPORTS_NX_ROLE = "supports-nx-role";
    public static final String PROP_FASTWILDCARDS = "fast-wildcards";
    public static final String PROP_REQUIRES_L3_MATCH = "requires-l3-match";
    public static final String PROP_SUPPORTS_OFPP_TABLE = "supports-ofpp-table";
    public static final String PROP_SUPPORTS_OFPP_FLOOD = "supports-ofpp-flood";
    public static final String PROP_SUPPORTS_NETMASK_TBL = "supports-netmask-table";
    public static final String PROP_SUPPORTS_BSN_SET_TUNNEL_DST_ACTION =
            "supports-set-tunnel-dst-action";
    public static final String PROP_SUPPORTS_NX_TTL_DECREMENT = "supports-nx-ttl-decrement";
    public enum SwitchStatus {
       HANDSHAKE(false),
       SLAVE(true),
       MASTER(true),
       QUARANTINED(false),
       DISCONNECTED(false);
       private final boolean visible;
       SwitchStatus(boolean visible) {
        this.visible = visible;
       }
       public boolean isVisible() {
            return visible;
       }
       public boolean isControllable() {
            return this == MASTER;
       }
    }
    SwitchStatus getStatus();
    long getBuffers();
    void disconnect();
    Set<OFActionType> getActions();
    Set<OFCapabilities> getCapabilities();
    Collection<TableId> getTables();
    SwitchDescription getSwitchDescription();
    SocketAddress getInetAddress();
    Collection<OFPortDesc> getEnabledPorts();
    Collection<OFPort> getEnabledPortNumbers();
    OFPortDesc getPort(OFPort portNumber);
    OFPortDesc getPort(String portName);
    Collection<OFPortDesc> getPorts();
    Collection<OFPortDesc> getSortedPorts();
    boolean portEnabled(OFPort portNumber);
    boolean portEnabled(String portName);
    boolean isConnected();
    Date getConnectedSince();
    DatapathId getId();
    Map<Object, Object> getAttributes();
    boolean isActive();
    OFControllerRole getControllerRole();
    boolean hasAttribute(String name);
    Object getAttribute(String name);
    boolean attributeEquals(String name, Object other);
    void setAttribute(String name, Object value);
    Object removeAttribute(String name);
    OFFactory getOFFactory();
    ImmutableList<IOFConnection> getConnections();
    boolean write(OFMessage m, LogicalOFMessageCategory category);
    Iterable<OFMessage> write(Iterable<OFMessage> msglist, LogicalOFMessageCategory category);
    OFConnection getConnectionByCategory(LogicalOFMessageCategory category);
    <REPLY extends OFStatsReply> ListenableFuture<List<REPLY>> writeStatsRequest(OFStatsRequest<REPLY> request, LogicalOFMessageCategory category);
    <R extends OFMessage> ListenableFuture<R> writeRequest(OFRequest<R> request, LogicalOFMessageCategory category);
    public TableFeatures getTableFeatures(TableId table);
	short getNumTables();
	public U64 getLatency();
}