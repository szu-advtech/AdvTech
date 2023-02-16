package net.floodlightcontroller.util;
import java.net.SocketAddress;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.floodlightcontroller.core.IOFConnection;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.SwitchDescription;
import net.floodlightcontroller.core.internal.OFConnection;
import net.floodlightcontroller.core.internal.TableFeatures;
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
public class OFMessageDamperMockSwitch implements IOFSwitch {
    OFMessage writtenMessage;
    public OFMessageDamperMockSwitch() {
        reset();
    }
    public void reset() {
        writtenMessage = null;
    }
    public void assertMessageWasWritten(OFMessage expected) {
        assertNotNull("No OFMessage was written", writtenMessage);
        assertEquals(expected, writtenMessage);
    }
    public void assertNoMessageWritten() {
        assertNull("OFMessage was written but didn't expect one",
                      writtenMessage);
    }
    @Override
    public boolean portEnabled(String portName) {
        assertTrue("Unexpected method call", false);
        return false;
    }
    @Override
    public DatapathId getId() {
        assertTrue("Unexpected method call", false);
        return DatapathId.NONE;
    }
    @Override
    public SocketAddress getInetAddress() {
        assertTrue("Unexpected method call", false);
        return null;
    }
    @Override
    public Map<Object, Object> getAttributes() {
        assertTrue("Unexpected method call", false);
        return null;
    }
    @Override
    public Date getConnectedSince() {
        assertTrue("Unexpected method call", false);
        return null;
    }
    @Override
    public boolean isConnected() {
        assertTrue("Unexpected method call", false);
        return false;
    }
    @Override
    public boolean hasAttribute(String name) {
        assertTrue("Unexpected method call", false);
        return false;
    }
    @Override
    public Object getAttribute(String name) {
        assertTrue("Unexpected method call", false);
        return null;
    }
    @Override
    public void setAttribute(String name, Object value) {
        assertTrue("Unexpected method call", false);
    }
    @Override
    public Object removeAttribute(String name) {
        assertTrue("Unexpected method call", false);
        return null;
    }
    @Override
    public long getBuffers() {
        fail("Unexpected method call");
        return 0;
    }
    @Override
    public Set<OFActionType> getActions() {
        fail("Unexpected method call");
        return null;
    }
    @Override
    public Set<OFCapabilities> getCapabilities() {
        fail("Unexpected method call");
        return null;
    }
    @Override
    public short getNumTables() {
        fail("Unexpected method call");
        return 0;
    }
    @Override
    public Collection<TableId> getTables() {
    	fail("Unexpected method call");
    	return null;
    }
    @Override
    public boolean attributeEquals(String name, Object other) {
        fail("Unexpected method call");
        return false;
    }
    @Override
    public boolean isActive() {
        fail("Unexpected method call");
    }
	@Override
	public boolean write(OFMessage m) {
		writtenMessage = m;
		return true;
	}
	@Override
	public Collection<OFMessage> write(Iterable<OFMessage> msgList) {
		return Collections.emptyList();
	}
	@Override
	public <R extends OFMessage> ListenableFuture<R> writeRequest(
			OFRequest<R> request) {
		return null;
	}
	@Override
	public <REPLY extends OFStatsReply> ListenableFuture<List<REPLY>> writeStatsRequest(
			OFStatsRequest<REPLY> request) {
		return null;
	}
	@Override
	public SwitchStatus getStatus() {
		return null;
	}
	@Override
	public void disconnect() {
	}
	@Override
	public SwitchDescription getSwitchDescription() {
		return null;
	}
	@Override
	public OFPortDesc getPort(OFPort portNumber) {
		return null;
	}
	@Override
	public Collection<OFPortDesc> getSortedPorts() {
		return null;
	}
	@Override
	public boolean portEnabled(OFPort portNumber) {
		return false;
	}
	@Override
	public OFControllerRole getControllerRole() {
		return null;
	}
	@Override
	public OFFactory getOFFactory() {
		return null;
	}
	@Override
	public ImmutableList<IOFConnection> getConnections() {
		return null;
	}
	@Override
	public boolean write(OFMessage m, LogicalOFMessageCategory category) {
		return true;
	}
	@Override
	public Collection<OFMessage> write(Iterable<OFMessage> msgList,
			LogicalOFMessageCategory category) {
		return Collections.emptyList();
	}
	@Override
	public OFConnection getConnectionByCategory(
			LogicalOFMessageCategory category) {
		return null;
	}
	@Override
	public <REPLY extends OFStatsReply> ListenableFuture<List<REPLY>> writeStatsRequest(
			OFStatsRequest<REPLY> request, LogicalOFMessageCategory category) {
		return null;
	}
	@Override
	public <R extends OFMessage> ListenableFuture<R> writeRequest(
			OFRequest<R> request, LogicalOFMessageCategory category) {
		return null;
	}
	@Override
	public Collection<OFPortDesc> getEnabledPorts() {
		return null;
	}
	@Override
	public Collection<OFPort> getEnabledPortNumbers() {
		return null;
	}
	@Override
	public OFPortDesc getPort(String portName) {
		return null;
	}
	@Override
	public Collection<OFPortDesc> getPorts() {
		return null;
	}
	@Override
	public TableFeatures getTableFeatures(TableId table) {
		return null;
	}
	@Override
	public U64 getLatency() {
		return null;
	}
}
