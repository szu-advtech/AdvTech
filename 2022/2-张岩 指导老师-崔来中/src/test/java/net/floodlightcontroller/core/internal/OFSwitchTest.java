package net.floodlightcontroller.core.internal;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.SwitchDriverSubHandshakeAlreadyStarted;
import net.floodlightcontroller.core.SwitchDriverSubHandshakeCompleted;
import net.floodlightcontroller.core.SwitchDriverSubHandshakeNotStarted;
import net.floodlightcontroller.core.util.URIUtil;
import org.projectfloodlight.openflow.protocol.OFBsnControllerConnection;
import org.projectfloodlight.openflow.protocol.OFBsnControllerConnectionState;
import org.projectfloodlight.openflow.protocol.OFBsnControllerConnectionsReply;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFNiciraControllerRole;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFAuxId;
public class OFSwitchTest {
	protected OFSwitch sw;
	protected OFFactory factory = OFFactories.getFactory(OFVersion.OF_13);
	@Before
	public void setUp() throws Exception {
		MockOFConnection mockConnection = new MockOFConnection(DatapathId.of(1), OFAuxId.MAIN);
		sw = new OFSwitch(mockConnection, OFFactories.getFactory(OFVersion.OF_10),
				EasyMock.createMock(IOFSwitchManager.class), DatapathId.of(1));
	}
	@Test
	public void testSetHARoleReply() {
		sw.setControllerRole(OFControllerRole.ROLE_MASTER);
		assertEquals(OFControllerRole.ROLE_MASTER, sw.getControllerRole());
		sw.setControllerRole(OFControllerRole.ROLE_EQUAL);
		assertEquals(OFControllerRole.ROLE_EQUAL, sw.getControllerRole());
		sw.setControllerRole(OFControllerRole.ROLE_SLAVE);
		assertEquals(OFControllerRole.ROLE_SLAVE, sw.getControllerRole());
	}
	@Test
	public void testSubHandshake() {
		OFFactory factory = OFFactories.getFactory(OFVersion.OF_10);
		OFMessage m = factory.buildNiciraControllerRoleReply()
				.setXid(1)
				.setRole(OFNiciraControllerRole.ROLE_MASTER)
				.build();
		try {
			sw.processDriverHandshakeMessage(m);
			fail("expected exception not thrown");
		try {
			sw.isDriverHandshakeComplete();
			fail("expected exception not thrown");
		sw.startDriverHandshake();
		assertTrue("Handshake should be complete",
				sw.isDriverHandshakeComplete());
		try {
			sw.processDriverHandshakeMessage(m);
			fail("expected exception not thrown");
		try {
			sw.startDriverHandshake();
			fail("Expected exception not thrown");
	}
	public void updateControllerConnections(IOFSwitchBackend sw, OFControllerRole role1, OFBsnControllerConnectionState state1, String uri1
			,  OFControllerRole role2, OFBsnControllerConnectionState state2, String uri2) {
		OFBsnControllerConnection connection1 = factory.buildBsnControllerConnection()
				.setAuxiliaryId(OFAuxId.MAIN)
				.setRole(role1)
				.setState(state1)
				.setUri(uri1)
				.build();
		OFBsnControllerConnection connection2 = factory.buildBsnControllerConnection()
				.setAuxiliaryId(OFAuxId.MAIN)
				.setRole(role2)
				.setState(state2)
				.setUri(uri2)
				.build();
		List<OFBsnControllerConnection> connections = new ArrayList<OFBsnControllerConnection>();
		connections.add(connection1);
		connections.add(connection2);
		OFBsnControllerConnectionsReply reply = factory.buildBsnControllerConnectionsReply()
				.setConnections(connections)
				.build();
		sw.updateControllerConnections(reply);
	}
	@Test
	public void testHasAnotherMaster() {
		URI cokeUri = URIUtil.createURI("1.2.3.4", 6653);
		InetSocketAddress address = (InetSocketAddress) sw.getConnection(OFAuxId.MAIN).getLocalInetAddress();
		URI pepsiUri = URIUtil.createURI(address.getHostName(), address.getPort());
		updateControllerConnections(sw, OFControllerRole.ROLE_SLAVE, OFBsnControllerConnectionState.BSN_CONTROLLER_CONNECTION_STATE_CONNECTED, cokeUri.toString(),
				OFControllerRole.ROLE_MASTER, OFBsnControllerConnectionState.BSN_CONTROLLER_CONNECTION_STATE_CONNECTED, pepsiUri.toString());
		assertFalse(sw.hasAnotherMaster());
		updateControllerConnections(sw, OFControllerRole.ROLE_MASTER, OFBsnControllerConnectionState.BSN_CONTROLLER_CONNECTION_STATE_CONNECTED, cokeUri.toString(),
				OFControllerRole.ROLE_SLAVE, OFBsnControllerConnectionState.BSN_CONTROLLER_CONNECTION_STATE_CONNECTED, pepsiUri.toString());
		assertTrue(sw.hasAnotherMaster());
	}
}
