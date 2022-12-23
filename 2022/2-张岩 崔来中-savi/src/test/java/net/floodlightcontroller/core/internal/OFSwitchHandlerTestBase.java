package net.floodlightcontroller.core.internal;
import static org.easymock.EasyMock.anyLong;
import static org.easymock.EasyMock.anyObject;
import static org.easymock.EasyMock.createMock;
import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.expectLastCall;
import static org.easymock.EasyMock.replay;
import static org.easymock.EasyMock.reset;
import static org.easymock.EasyMock.resetToStrict;
import static org.easymock.EasyMock.verify;
import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.CoreMatchers.notNullValue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.easymock.EasyMock;
import org.hamcrest.CoreMatchers;
import org.hamcrest.Matchers;
import io.netty.util.Timeout;
import io.netty.util.Timer;
import io.netty.util.TimerTask;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitch.SwitchStatus;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.PortChangeEvent;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.internal.OFSwitchAppHandshakePlugin.PluginResultType;
import net.floodlightcontroller.core.internal.OFSwitchHandshakeHandler.QuarantineState;
import net.floodlightcontroller.core.internal.OFSwitchHandshakeHandler.WaitAppHandshakeState;
import net.floodlightcontroller.debugcounter.DebugCounterServiceImpl;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import org.projectfloodlight.openflow.protocol.OFBadActionCode;
import org.projectfloodlight.openflow.protocol.OFBadRequestCode;
import org.projectfloodlight.openflow.protocol.OFBarrierReply;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFDescStatsReply;
import org.projectfloodlight.openflow.protocol.OFErrorMsg;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFGetConfigReply;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketInReason;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFPortReason;
import org.projectfloodlight.openflow.protocol.OFPortStatus;
import org.projectfloodlight.openflow.protocol.OFSetConfig;
import org.projectfloodlight.openflow.protocol.OFStatsRequest;
import org.projectfloodlight.openflow.protocol.OFStatsType;
import org.projectfloodlight.openflow.protocol.OFTableFeatureProp;
import org.projectfloodlight.openflow.protocol.OFTableFeaturesStatsReply;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U32;
import org.projectfloodlight.openflow.types.U64;
import net.floodlightcontroller.util.LinkedHashSetWrapper;
import net.floodlightcontroller.util.OrderedCollection;
import com.google.common.collect.ImmutableList;
public abstract class OFSwitchHandlerTestBase {
	protected static final DatapathId dpid = DatapathId.of(0x42L);
	protected IOFSwitchManager switchManager;
	protected RoleManager roleManager;
	private IDebugCounterService debugCounterService;
	protected OFSwitchHandshakeHandler switchHandler;
	protected MockOFConnection connection;
	protected final OFFactory factory = getFactory();
	protected OFFeaturesReply featuresReply;
	protected List<IAppHandshakePluginFactory> plugins;
	private HashSet<Long> seenXids = null;
	protected IOFSwitchBackend sw;
	private Timer timer;
	private TestHandshakePlugin handshakePlugin;
	private class TestHandshakePlugin extends OFSwitchAppHandshakePlugin {
		protected TestHandshakePlugin(PluginResult defaultResult, int timeoutS) {
			super(defaultResult, timeoutS);
		}
		@Override
		protected void processOFMessage(OFMessage m) {
		}
		@Override
		protected void enterPlugin() {
		}
	}
	public void setUpFeaturesReply() {
		getFeaturesReply();
		this.featuresReply = getFeaturesReply();
		IAppHandshakePluginFactory factory = createMock(IAppHandshakePluginFactory.class);
		PluginResult result = new PluginResult(PluginResultType.QUARANTINE, "test quarantine");
		handshakePlugin = new TestHandshakePlugin(result, 5);
		expect(factory.createPlugin()).andReturn(handshakePlugin).anyTimes();
		replay(factory);
		plugins = ImmutableList.of(factory);
	}
	@Before
	public void setUp() throws Exception {
		setUpFeaturesReply(); 
		switchManager = createMock(IOFSwitchManager.class);
		roleManager = createMock(RoleManager.class);
		sw = createMock(IOFSwitchBackend.class);
		timer = createMock(Timer.class);
		expect(timer.newTimeout(anyObject(TimerTask.class), anyLong(), anyObject(TimeUnit.class))).andReturn(EasyMock.createNiceMock(Timeout.class));
		replay(timer);
		seenXids = null;
		debugCounterService = new DebugCounterServiceImpl();
		SwitchManagerCounters counters =
				new SwitchManagerCounters(debugCounterService);
		expect(switchManager.getCounters()).andReturn(counters).anyTimes();
		replay(switchManager);
		connection = new MockOFConnection(featuresReply.getDatapathId(), OFAuxId.MAIN);
		switchHandler = new OFSwitchHandshakeHandler(connection, featuresReply, switchManager, roleManager, timer);
		replay(sw);
	}
	@After
	public void tearDown() {
		verifyAll();
	}
	private void verifyAll() {
		assertThat("Unexpected messages have been captured",
				connection.getMessages(),
				Matchers.empty());
		verify(sw);
	}
	void verifyUniqueXids(OFMessage... msgs) {
		verifyUniqueXids(Arrays.asList(msgs));
	}
	void verifyUniqueXids(List<OFMessage> msgs) {
		if (seenXids == null)
			seenXids = new HashSet<Long>();
		for (OFMessage m: msgs)  {
			long xid = m.getXid();
			assertTrue("Xid in messags is 0", xid != 0);
			assertFalse("Xid " + xid + " has already been used",
					seenXids.contains(xid));
			seenXids.add(xid);
		}
	}
	public abstract OFFactory getFactory();
	abstract OFFeaturesReply getFeaturesReply();
	abstract Class<?> getRoleRequestClass();
	public abstract void verifyRoleRequest(OFMessage m,
			OFControllerRole expectedRole);
	protected abstract OFMessage getRoleReply(long xid, OFControllerRole role);
	abstract void moveToPreConfigReply() throws Exception;
	@Test
	public abstract void moveToWaitAppHandshakeState() throws Exception;
	@Test
	public abstract void moveToWaitSwitchDriverSubHandshake() throws Exception;
	@Test
	public abstract void moveToWaitInitialRole() throws Exception;
        This occurs upon creation of the switch handler
	@Test
	public void testInitState() throws Exception {
		assertThat(connection.getListener(), notNullValue());
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.InitState.class));
	}
	@Test
	public void moveToWaitConfigReply() throws Exception {
		moveToPreConfigReply();
		List<OFMessage> msgs = connection.getMessages();
		assertEquals(3, msgs.size());
		assertEquals(OFType.SET_CONFIG, msgs.get(0).getType());
		OFSetConfig sc = (OFSetConfig)msgs.get(0);
		assertEquals(0xffff, sc.getMissSendLen());
		assertEquals(OFType.BARRIER_REQUEST, msgs.get(1).getType());
		assertEquals(OFType.GET_CONFIG_REQUEST, msgs.get(2).getType());
		verifyUniqueXids(msgs);
		msgs.clear();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitConfigReplyState.class));
		verifyAll();
	}
	@Test
	public void moveToWaitDescriptionStatReply() throws Exception {
		moveToWaitConfigReply();
		connection.clearMessages();
		OFGetConfigReply cr = factory.buildGetConfigReply()
				.setMissSendLen(0xFFFF)
				.build();
		switchHandler.processOFMessage(cr);
		OFMessage msg = connection.retrieveMessage();
		assertEquals(OFType.STATS_REQUEST, msg.getType());
		OFStatsRequest<?> sr = (OFStatsRequest<?>)msg;
		assertEquals(OFStatsType.DESC, sr.getStatsType());
		verifyUniqueXids(msg);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitDescriptionStatReplyState.class));
	}
	protected OFDescStatsReply createDescriptionStatsReply() {
		OFDescStatsReply statsReply = factory.buildDescStatsReply()
				.setDpDesc("Datapath Description")
				.setHwDesc("Hardware Description")
				.setMfrDesc("Manufacturer Description")
				.setSwDesc("Software Description")
				.setSerialNum("Serial Number")
				.build();
		return statsReply;
	}
	protected OFTableFeaturesStatsReply createTableFeaturesStatsReply() {
		OFTableFeaturesStatsReply statsReply = factory.buildTableFeaturesStatsReply()
				.setEntries(Collections.singletonList(factory.buildTableFeatures()
						.setConfig(0)
						.setMaxEntries(100)
						.setMetadataMatch(U64.NO_MASK)
						.setMetadataWrite(U64.NO_MASK)
						.setName("MyTable")
						.setTableId(TableId.of(1))
						.setProperties(Collections.singletonList((OFTableFeatureProp)factory.buildTableFeaturePropMatch()
								.setOxmIds(Collections.singletonList(U32.of(100)))
								.build())
								).build()
						)
						).build();
		return statsReply;
	}
	protected void setupSwitchForInstantiationWithReset()
			throws Exception {
		reset(sw);
		sw.setFeaturesReply(featuresReply);
		expectLastCall().once();
	}
	@Test
	public void moveQuarantine() throws Exception {
		moveToWaitAppHandshakeState();
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.QUARANTINED);
		expectLastCall().once();
		replay(switchManager);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(WaitAppHandshakeState.class));
		WaitAppHandshakeState state = (WaitAppHandshakeState) switchHandler.getStateForTesting();
		assertThat(state.getCurrentPlugin(), CoreMatchers.<OFSwitchAppHandshakePlugin>equalTo(handshakePlugin));
		reset(sw);
		expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE);
		sw.setStatus(SwitchStatus.QUARANTINED);
		expectLastCall().once();
		replay(sw);
		PluginResult result = new PluginResult(PluginResultType.QUARANTINE, "test quarantine");
		handshakePlugin.exitPlugin(result);
		assertThat(switchHandler.getStateForTesting(),
				CoreMatchers.instanceOf(QuarantineState.class));
		verify(switchManager);
	}
	@Test
	public void failedAppHandshake() throws Exception {
		moveToWaitAppHandshakeState();
		assertThat(switchHandler.getStateForTesting(),
				CoreMatchers.instanceOf(WaitAppHandshakeState.class));
		WaitAppHandshakeState state = (WaitAppHandshakeState) switchHandler.getStateForTesting();
		assertThat(state.getCurrentPlugin(), CoreMatchers.<OFSwitchAppHandshakePlugin>equalTo(handshakePlugin));
		PluginResult result = new PluginResult(PluginResultType.DISCONNECT);
		handshakePlugin.exitPlugin(result);
		assertThat(connection.isConnected(), equalTo(false));
	}
	@Test
	public void validAppHandshakePluginReason() throws Exception {
		try{
			new PluginResult(PluginResultType.QUARANTINE,"This should not cause an exception");
		}catch(IllegalStateException e) {
			fail("This should cause an illegal state exception");
		}
	}
	@Test
	public void invalidAppHandshakePluginReason() throws Exception {
		try{
			new PluginResult(PluginResultType.CONTINUE,"This should cause an exception");
			fail("This should cause an illegal state exception");
		try{
			new PluginResult(PluginResultType.DISCONNECT,"This should cause an exception");
			fail("This should cause an illegal state exception");
	}
	@Test
	public void testSwitchDriverSubHandshake()
			throws Exception {
		moveToWaitSwitchDriverSubHandshake();
		Match match = factory.buildMatch().build();
		OFMessage m = factory.buildFlowRemoved().setMatch(match).build();
		resetToStrict(sw);
		sw.processDriverHandshakeMessage(m);
		expectLastCall().once();
		expect(sw.isDriverHandshakeComplete()).andReturn(true).once();
		replay(sw);
		switchHandler.processOFMessage(m);
		assertThat(switchHandler.getStateForTesting(),
				CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitAppHandshakeState.class));
		assertThat("Unexpected message captured", connection.getMessages(), Matchers.empty());
		verify(sw);
	}
	@Test
	public void testWaitDescriptionReplyState() throws Exception {
		moveToWaitInitialRole();
	}
	private long setupSwitchSendRoleRequestAndVerify(Boolean supportsNxRole,
			OFControllerRole role) throws IOException {
		assertTrue("This internal test helper method most not be called " +
				"with supportsNxRole==false. Test setup broken",
				supportsNxRole == null || supportsNxRole == true);
		reset(sw);
		expect(sw.getAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE))
		.andReturn(supportsNxRole).atLeastOnce();
		replay(sw);
		switchHandler.sendRoleRequest(role);
		OFMessage msg = connection.retrieveMessage();
		verifyRoleRequest(msg, role);
		verify(sw);
		return msg.getXid();
	}
	@SuppressWarnings("unchecked")
	private void setupSwitchRoleChangeUnsupported(int xid,
			OFControllerRole role) {
		SwitchStatus newStatus = role != OFControllerRole.ROLE_SLAVE ? SwitchStatus.MASTER : SwitchStatus.SLAVE;
		boolean supportsNxRole = false;
		verify(switchManager);
		reset(sw, switchManager);
		expect(sw.getAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE))
		.andReturn(supportsNxRole).atLeastOnce();
		expect(sw.getOFFactory()).andReturn(factory).anyTimes();
		expect(sw.write(anyObject(OFMessage.class))).andReturn(true).anyTimes();
		expect(sw.write(anyObject(Iterable.class))).andReturn(Collections.EMPTY_LIST).anyTimes();
		expect(sw.getNumTables()).andStubReturn((short)0);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, supportsNxRole);
		expectLastCall().anyTimes();
		if (SwitchStatus.MASTER == newStatus) {
			if (factory.getVersion().compareTo(OFVersion.OF_13) >= 0) {
				expect(sw.getTables()).andReturn(Collections.EMPTY_LIST).once();
				expect(sw.getTableFeatures(TableId.ZERO)).andReturn(TableFeatures.of(createTableFeaturesStatsReply().getEntries().get(0))).anyTimes();
			}
		}
		sw.setControllerRole(role);
		expectLastCall().once();
		if (role == OFControllerRole.ROLE_SLAVE) {
			sw.disconnect();
			expectLastCall().once();
		} else {
			expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE).once();
			sw.setStatus(newStatus);
			expectLastCall().once();
			switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, newStatus);
		}
		replay(sw, switchManager);
		switchHandler.sendRoleRequest(role);
		OFBarrierReply br = getFactory().buildBarrierReply()
				.build();
		switchHandler.processOFMessage(br);
		verify(sw, switchManager);
	}
	private OFMessage getBadRequestErrorMessage(OFBadRequestCode code, long xid) {
		OFErrorMsg msg = factory.errorMsgs().buildBadRequestErrorMsg()
				.setXid(xid)
				.setCode(code)
				.build();
		return msg;
	}
	private OFMessage getBadActionErrorMessage(OFBadActionCode code, long xid) {
		OFErrorMsg msg = factory.errorMsgs().buildBadActionErrorMsg()
				.setXid(xid)
				.setCode(code)
				.build();
		return msg;
	}
	@SuppressWarnings("unchecked")
	@Test
	public void testInitialMoveToMasterWithRole() throws Exception {
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		long xid = setupSwitchSendRoleRequestAndVerify(null, OFControllerRole.ROLE_MASTER);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		reset(sw);
		expect(sw.getOFFactory()).andReturn(factory).anyTimes();
		expect(sw.write(anyObject(OFMessage.class))).andReturn(true).anyTimes();
		expect(sw.write(anyObject(Iterable.class))).andReturn(Collections.EMPTY_LIST).anyTimes();
		expect(sw.getTables()).andStubReturn(Collections.EMPTY_LIST);
		expect(sw.getNumTables()).andStubReturn((short) 0);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, true);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_MASTER);
		expectLastCall().once();
		expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE).once();
		sw.setStatus(SwitchStatus.MASTER);
		expectLastCall().once();
		if (factory.getVersion().compareTo(OFVersion.OF_13) >= 0) {
			expect(sw.getTableFeatures(TableId.ZERO)).andReturn(TableFeatures.of(createTableFeaturesStatsReply().getEntries().get(0))).anyTimes();
		}
		replay(sw);
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
		expectLastCall().once();
		replay(switchManager);
		OFMessage reply = getRoleReply(xid, OFControllerRole.ROLE_MASTER);
		switchHandler.processOFMessage(reply);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.MasterState.class));
	}
	@Test
	public void testInitialMoveToSlaveWithRole() throws Exception {
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		long xid = setupSwitchSendRoleRequestAndVerify(null, OFControllerRole.ROLE_SLAVE);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		reset(sw);
		sw.setAttribute(IOFSwitchBackend.SWITCH_SUPPORTS_NX_ROLE, true);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_SLAVE);
		expectLastCall().once();
		expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE).once();
		sw.setStatus(SwitchStatus.SLAVE);
		expectLastCall().once();
		replay(sw);
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.SLAVE);
		expectLastCall().once();
		replay(switchManager);
		OFMessage reply = getRoleReply(xid, OFControllerRole.ROLE_SLAVE);
		switchHandler.processOFMessage(reply);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.SlaveState.class));
	}
	@SuppressWarnings("unchecked")
	@Test
	public void testInitialMoveToMasterNoRole() throws Exception {
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		long xid = setupSwitchSendRoleRequestAndVerify(null, OFControllerRole.ROLE_MASTER);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		reset(sw);
		expect(sw.getOFFactory()).andReturn(factory).anyTimes();
		expect(sw.write(anyObject(OFMessage.class))).andReturn(true).anyTimes();
		expect(sw.write(anyObject(Iterable.class))).andReturn(Collections.EMPTY_LIST).anyTimes();
		expect(sw.getNumTables()).andStubReturn((short)0);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, false);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_MASTER);
		expectLastCall().once();
		expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE).once();
		sw.setStatus(SwitchStatus.MASTER);
		expectLastCall().once();
		if (factory.getVersion().compareTo(OFVersion.OF_13) >= 0) {
			expect(sw.getTables()).andReturn(Collections.EMPTY_LIST).once();
			expect(sw.getTableFeatures(TableId.ZERO)).andReturn(TableFeatures.of(createTableFeaturesStatsReply().getEntries().get(0))).anyTimes();
		}
		replay(sw);
		OFMessage err = getBadActionErrorMessage(OFBadActionCode.BAD_TYPE, xid+1);
		switchHandler.processOFMessage(err);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		err = getBadRequestErrorMessage(OFBadRequestCode.BAD_EXPERIMENTER, xid);
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
		expectLastCall().once();
		replay(switchManager);
		switchHandler.processOFMessage(err);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.MasterState.class));
	}
	@SuppressWarnings("unchecked")
	@Test
	public void testInitialMoveToMasterTimeout() throws Exception {
		int timeout = 50;
		switchHandler.useRoleChangerWithOtherTimeoutForTesting(timeout);
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		reset(sw);
		expect(sw.getOFFactory()).andReturn(factory).anyTimes();
		expect(sw.write(anyObject(OFMessage.class))).andReturn(true).anyTimes();
		expect(sw.write(anyObject(Iterable.class))).andReturn(Collections.EMPTY_LIST).anyTimes();
		expect(sw.getNumTables()).andStubReturn((short)0);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, false);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_MASTER);
		expectLastCall().once();
		expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE).once();
		sw.setStatus(SwitchStatus.MASTER);
		expectLastCall().once();
		if (factory.getVersion().compareTo(OFVersion.OF_13) >= 0) {
			expect(sw.getTables()).andReturn(Collections.EMPTY_LIST).once();
			expect(sw.getTableFeatures(TableId.ZERO)).andReturn(TableFeatures.of(createTableFeaturesStatsReply().getEntries().get(0))).anyTimes();
		}
		replay(sw);
		OFMessage m = factory.barrierReply();
		Thread.sleep(timeout + 5);
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
		expectLastCall().once();
		replay(switchManager);
		switchHandler.processOFMessage(m);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.MasterState.class));
	}
	@Test
	public void testInitialMoveToSlaveNoRole() throws Exception {
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		long xid = setupSwitchSendRoleRequestAndVerify(null, OFControllerRole.ROLE_SLAVE);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		reset(sw);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, false);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_SLAVE);
		expectLastCall().once();
		expectLastCall().once();
		replay(sw);
		OFMessage err = getBadActionErrorMessage(OFBadActionCode.BAD_TYPE, xid+1);
		switchHandler.processOFMessage(err);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		err = getBadRequestErrorMessage(OFBadRequestCode.BAD_EXPERIMENTER, xid);
		switchHandler.processOFMessage(err);
	}
	@Test
	public void testInitialMoveToSlaveTimeout() throws Exception {
		int timeout = 50;
		switchHandler.useRoleChangerWithOtherTimeoutForTesting(timeout);
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		setupSwitchSendRoleRequestAndVerify(null, OFControllerRole.ROLE_SLAVE);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		reset(sw);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, false);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_SLAVE);
		expectLastCall().once();
		expectLastCall().once();
		replay(sw);
		OFMessage m = factory.buildBarrierReply().build();
		Thread.sleep(timeout+5);
		switchHandler.processOFMessage(m);
	}
	@Test
	public void testNoRoleInitialToMasterToSlave() throws Exception {
		int xid = 46;
		testInitialMoveToMasterNoRole();
		assertThat(switchHandler.getStateForTesting(),
				CoreMatchers.instanceOf(OFSwitchHandshakeHandler.MasterState.class));
		assertThat("Unexpected messages have been captured",
				connection.getMessages(),
				Matchers.empty());
		setupSwitchRoleChangeUnsupported(xid, OFControllerRole.ROLE_MASTER);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.MasterState.class));
		assertThat("Unexpected messages have been captured",
				connection.getMessages(),
				Matchers.empty());
		setupSwitchRoleChangeUnsupported(xid, OFControllerRole.ROLE_SLAVE);
		assertThat(connection.isConnected(), equalTo(false));
		assertThat("Unexpected messages have been captured",
				connection.getMessages(),
				Matchers.empty());
	}
	@SuppressWarnings("unchecked")
	public void changeRoleToMasterWithRequest() throws Exception {
		assertTrue("This method can only be called when handler is in " +
				"MASTER or SLAVE role", switchHandler.isHandshakeComplete());
		long xid = setupSwitchSendRoleRequestAndVerify(true, OFControllerRole.ROLE_MASTER);
		reset(sw);
		expect(sw.getOFFactory()).andReturn(factory).anyTimes();
		expect(sw.write(anyObject(OFMessage.class))).andReturn(true).anyTimes();
		expect(sw.write(anyObject(Iterable.class))).andReturn(Collections.EMPTY_LIST).anyTimes();
		expect(sw.getNumTables()).andStubReturn((short)0);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, true);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_MASTER);
		expectLastCall().once();
		expect(sw.getStatus()).andReturn(SwitchStatus.HANDSHAKE).once();
		sw.setStatus(SwitchStatus.MASTER);
		expectLastCall().once();
		expect(sw.getTables()).andReturn(Collections.EMPTY_LIST).once();
		replay(sw);
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
		expectLastCall().once();
		replay(switchManager);
		OFMessage reply = getRoleReply(xid, OFControllerRole.ROLE_MASTER);
		switchHandler.processOFMessage(reply);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.MasterState.class));
	}
	public void changeRoleToSlaveWithRequest() throws Exception {
		assertTrue("This method can only be called when handler is in " +
				"MASTER or SLAVE role", switchHandler.isHandshakeComplete());
		long xid = setupSwitchSendRoleRequestAndVerify(true, OFControllerRole.ROLE_SLAVE);
		reset(sw);
		sw.setAttribute(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, true);
		expectLastCall().once();
		sw.setControllerRole(OFControllerRole.ROLE_SLAVE);
		expectLastCall().once();
		expect(sw.getStatus()).andReturn(SwitchStatus.MASTER).once();
		sw.setStatus(SwitchStatus.SLAVE);
		expectLastCall().once();
		replay(sw);
		reset(switchManager);
		switchManager.switchStatusChanged(sw, SwitchStatus.MASTER, SwitchStatus.SLAVE);
		expectLastCall().once();
		replay(switchManager);
		OFMessage reply = getRoleReply(xid, OFControllerRole.ROLE_SLAVE);
		connection.getListener().messageReceived(connection, reply);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.SlaveState.class));
	}
	@Test
	public void testMultiRoleChange1() throws Exception {
		testInitialMoveToMasterWithRole();
		changeRoleToMasterWithRequest();
		changeRoleToSlaveWithRequest();
		changeRoleToSlaveWithRequest();
		changeRoleToMasterWithRequest();
		changeRoleToSlaveWithRequest();
	}
	@Test
	public void testMultiRoleChange2() throws Exception {
		testInitialMoveToSlaveWithRole();
		changeRoleToMasterWithRequest();
		changeRoleToSlaveWithRequest();
		changeRoleToSlaveWithRequest();
		changeRoleToMasterWithRequest();
		changeRoleToSlaveWithRequest();
	}
	@Test
	public void testInitialRoleChangeOtherError() throws Exception {
		moveToWaitInitialRole();
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		long xid = setupSwitchSendRoleRequestAndVerify(null, OFControllerRole.ROLE_MASTER);
		assertThat(switchHandler.getStateForTesting(), CoreMatchers.instanceOf(OFSwitchHandshakeHandler.WaitInitialRoleState.class));
		OFMessage err = getBadActionErrorMessage(OFBadActionCode.BAD_TYPE, xid);
		verifyExceptionCaptured(err, SwitchStateException.class);
	}
	@Test
	public void testMessageDispatchMaster() throws Exception {
		testInitialMoveToMasterWithRole();
		OFPacketIn pi = factory.buildPacketIn()
				.setReason(OFPacketInReason.NO_MATCH)
				.build();
		reset(switchManager);
		switchManager.handleMessage(sw, pi, null);
		expectLastCall().once();
		replay(switchManager);
		switchHandler.processOFMessage(pi);
	}
	@Test
	public void testPortStatusMessageMaster() throws Exception {
		DatapathId dpid = featuresReply.getDatapathId();
		testInitialMoveToMasterWithRole();
		OFPortDesc portDesc = factory.buildPortDesc()
				.setName("Port1")
				.setPortNo(OFPort.of(1))
				.build();
		OFPortStatus.Builder portStatusBuilder = factory.buildPortStatus()
				.setDesc(portDesc);
		OrderedCollection<PortChangeEvent> events =
				new LinkedHashSetWrapper<PortChangeEvent>();
		OFPortDesc.Builder pb = factory.buildPortDesc();
		OFPortDesc p1 = pb.setName("eth1").setPortNo(OFPort.of(1)).build();
		OFPortDesc p2 = pb.setName("eth2").setPortNo(OFPort.of(2)).build();
		OFPortDesc p3 = pb.setName("eth3").setPortNo(OFPort.of(3)).build();
		OFPortDesc p4 = pb.setName("eth4").setPortNo(OFPort.of(4)).build();
		OFPortDesc p5 = pb.setName("eth5").setPortNo(OFPort.of(5)).build();
		events.add(new PortChangeEvent(p1, PortChangeType.ADD));
		events.add(new PortChangeEvent(p2, PortChangeType.DELETE));
		events.add(new PortChangeEvent(p3, PortChangeType.UP));
		events.add(new PortChangeEvent(p4, PortChangeType.DOWN));
		events.add(new PortChangeEvent(p5, PortChangeType.OTHER_UPDATE));
		for (OFPortReason reason: OFPortReason.values()) {
			OFPortStatus portStatus = portStatusBuilder.setReason(reason).build();
			reset(sw);
			expect(sw.getId()).andReturn(dpid).anyTimes();
			expect(sw.processOFPortStatus(portStatus)).andReturn(events).once();
			replay(sw);
			reset(switchManager);
			switchManager.notifyPortChanged(sw, p1, PortChangeType.ADD);
			switchManager.notifyPortChanged(sw, p2, PortChangeType.DELETE);
			switchManager.notifyPortChanged(sw, p3, PortChangeType.UP);
			switchManager.notifyPortChanged(sw, p4, PortChangeType.DOWN);
			switchManager.notifyPortChanged(sw, p5, PortChangeType.OTHER_UPDATE);
			replay(switchManager);
			switchHandler.processOFMessage(portStatus);
			verify(sw);
		}
	}
	@Test
	public void testReassertMaster() throws Exception {
		testInitialMoveToMasterWithRole();
		OFMessage err = getBadRequestErrorMessage(OFBadRequestCode.EPERM, 42);
		reset(roleManager);
		roleManager.reassertRole(switchHandler, HARole.ACTIVE);
		expectLastCall().once();
		replay(roleManager);
		reset(switchManager);
		switchManager.handleMessage(sw, err, null);
		expectLastCall().once();
		replay(switchManager);
		switchHandler.processOFMessage(err);
		verify(sw);
	}
	void verifyExceptionCaptured(
			OFMessage err, Class<? extends Throwable> expectedExceptionClass) {
		Throwable caughtEx = null;
		try{
			switchHandler.processOFMessage(err);
		}
		catch(Exception e){
			caughtEx = e;
		}
		assertThat(caughtEx, CoreMatchers.instanceOf(expectedExceptionClass));
	}
	@Test
	public void testConnectionClosedBeforeHandshakeComplete() {
		reset(switchManager);
		switchManager.handshakeDisconnected(dpid);
		expectLastCall().once();
		replay(switchManager);
		switchHandler.connectionClosed(connection);
		verify(switchManager);
	}
	@Test
	public void testConnectionClosedAfterHandshakeComplete() throws Exception {
		testInitialMoveToMasterWithRole();
		reset(switchManager);
		switchManager.handshakeDisconnected(dpid);
		expectLastCall().once();
		switchManager.switchDisconnected(sw);
		expectLastCall().once();
		replay(switchManager);
		reset(sw);
		expect(sw.getStatus()).andReturn(SwitchStatus.DISCONNECTED).anyTimes();
		replay(sw);
		switchHandler.connectionClosed(connection);
		verify(switchManager);
		verify(sw);
	}
}
