package net.floodlightcontroller.staticflowentry;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.easymock.Capture;
import org.easymock.CaptureType;
import org.junit.Test;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.OFFlowModFlags;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.util.HexString;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.test.MockFloodlightProvider;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
import net.floodlightcontroller.test.FloodlightTestCase;
import net.floodlightcontroller.util.FlowModUtils;
import net.floodlightcontroller.util.MatchUtils;
import net.floodlightcontroller.util.OFMessageUtils;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.restserver.RestApiServer;
import net.floodlightcontroller.staticflowentry.StaticFlowEntryPusher;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.storage.memory.MemoryStorageSource;
public class StaticFlowTests extends FloodlightTestCase {
	static String TestSwitch1DPID = "00:00:00:00:00:00:00:01";
	static int TotalTestRules = 3;
	static OFFactory factory = OFFactories.getFactory(OFVersion.OF_13);
	static Map<String,Object> TestRule1;
	static OFFlowMod FlowMod1;
	static {
		FlowMod1 = factory.buildFlowModify().build();
		TestRule1 = new HashMap<String,Object>();
		TestRule1.put(COLUMN_NAME, "TestRule1");
		TestRule1.put(COLUMN_SWITCH, TestSwitch1DPID);
		Match match;
		TestRule1.put(COLUMN_DL_DST, "00:20:30:40:50:60");
		match = MatchUtils.fromString("eth_dst=00:20:30:40:50:60", factory.getVersion());
		List<OFAction> actions = new LinkedList<OFAction>();
		TestRule1.put(COLUMN_ACTIONS, "output=1");
		actions.add(factory.actions().output(OFPort.of(1), Integer.MAX_VALUE));
		FlowMod1 = FlowMod1.createBuilder().setMatch(match)
				.setActions(actions)
				.setFlags(Collections.singleton(OFFlowModFlags.SEND_FLOW_REM))
				.setBufferId(OFBufferId.NO_BUFFER)
				.setOutPort(OFPort.ANY)
				.setPriority(Integer.MAX_VALUE)
				.setXid(4)
				.build();
	}
	static Map<String,Object> TestRule2;
	static OFFlowMod FlowMod2;
	static {
		FlowMod2 = factory.buildFlowModify().build();
		TestRule2 = new HashMap<String,Object>();
		TestRule2.put(COLUMN_NAME, "TestRule2");
		TestRule2.put(COLUMN_SWITCH, TestSwitch1DPID);
		Match match;        
		TestRule2.put(COLUMN_DL_TYPE, "0x800");
		TestRule2.put(COLUMN_NW_DST, "192.168.1.0/24");
		match = MatchUtils.fromString("eth_type=0x800,ipv4_dst=192.168.1.0/24", factory.getVersion());
		List<OFAction> actions = new LinkedList<OFAction>();
		TestRule2.put(COLUMN_ACTIONS, "output=1");
		actions.add(factory.actions().output(OFPort.of(1), Integer.MAX_VALUE));
		FlowMod2 = FlowMod2.createBuilder().setMatch(match)
				.setActions(actions)
				.setFlags(Collections.singleton(OFFlowModFlags.SEND_FLOW_REM))
				.setBufferId(OFBufferId.NO_BUFFER)
				.setOutPort(OFPort.ANY)
				.setPriority(Integer.MAX_VALUE)
				.setXid(5)
				.build();
	}
	static Map<String,Object> TestRule3;
	static OFFlowMod FlowMod3;
	private StaticFlowEntryPusher staticFlowEntryPusher;
	private IOFSwitchService switchService;
	private IOFSwitch mockSwitch;
	private MockDebugCounterService debugCounterService;
	private Capture<OFMessage> writeCapture;
	private Capture<List<OFMessage>> writeCaptureList;
	private long dpid;
	private MemoryStorageSource storage;
	static {
		FlowMod3 = factory.buildFlowModify().build();
		TestRule3 = new HashMap<String,Object>();
		TestRule3.put(COLUMN_NAME, "TestRule3");
		TestRule3.put(COLUMN_SWITCH, TestSwitch1DPID);
		Match match;
		TestRule3.put(COLUMN_DL_DST, "00:20:30:40:50:60");
		TestRule3.put(COLUMN_DL_VLAN, 96);
		match = MatchUtils.fromString("eth_dst=00:20:30:40:50:60,eth_vlan_vid=96", factory.getVersion());
		TestRule3.put(COLUMN_ACTIONS, "output=controller");
		List<OFAction> actions = new LinkedList<OFAction>();
		actions.add(factory.actions().output(OFPort.CONTROLLER, Integer.MAX_VALUE));
		FlowMod3 = FlowMod3.createBuilder().setMatch(match)
				.setActions(actions)
		        .setFlags(Collections.singleton(OFFlowModFlags.SEND_FLOW_REM))
				.setBufferId(OFBufferId.NO_BUFFER)
				.setOutPort(OFPort.ANY)
				.setPriority(Integer.MAX_VALUE)
				.setXid(6)
				.build();
	}
	private void verifyFlowMod(OFFlowMod testFlowMod, OFFlowMod goodFlowMod) {
		verifyMatch(testFlowMod, goodFlowMod);
		verifyActions(testFlowMod, goodFlowMod);
		goodFlowMod = goodFlowMod.createBuilder().setCookie(testFlowMod.getCookie()).build();
		assertTrue(OFMessageUtils.equalsIgnoreXid(goodFlowMod, testFlowMod));
	}
	private void verifyMatch(OFFlowMod testFlowMod, OFFlowMod goodFlowMod) {
		assertEquals(goodFlowMod.getMatch(), testFlowMod.getMatch());
	}
	private void verifyActions(OFFlowMod testFlowMod, OFFlowMod goodFlowMod) {
		List<OFAction> goodActions = goodFlowMod.getActions();
		List<OFAction> testActions = testFlowMod.getActions();
		assertNotNull(goodActions);
		assertNotNull(testActions);
		assertEquals(goodActions.size(), testActions.size());
		for(int i = 0; i < goodActions.size(); i++) {
			assertEquals(goodActions.get(i), testActions.get(i));
		}
	}
	@Override
	public void setUp() throws Exception {
		super.setUp();
		debugCounterService = new MockDebugCounterService();
		staticFlowEntryPusher = new StaticFlowEntryPusher();
		switchService = getMockSwitchService();
		storage = new MemoryStorageSource();
		dpid = HexString.toLong(TestSwitch1DPID);
		mockSwitch = createNiceMock(IOFSwitch.class);
		writeCapture = new Capture<OFMessage>(CaptureType.ALL);
		writeCaptureList = new Capture<List<OFMessage>>(CaptureType.ALL);
		expect(mockSwitch.write(capture(writeCapture))).andReturn(true).anyTimes();
		expect(mockSwitch.write(capture(writeCaptureList))).andReturn(Collections.<OFMessage>emptyList()).anyTimes();
		expect(mockSwitch.getOFFactory()).andReturn(factory).anyTimes();
		replay(mockSwitch);
		FloodlightModuleContext fmc = new FloodlightModuleContext();
		fmc.addService(IStorageSourceService.class, storage);
		fmc.addService(IOFSwitchService.class, getMockSwitchService());
		fmc.addService(IDebugCounterService.class, debugCounterService);
		MockFloodlightProvider mockFloodlightProvider = getMockFloodlightProvider();
		Map<DatapathId, IOFSwitch> switchMap = new HashMap<DatapathId, IOFSwitch>();
		switchMap.put(DatapathId.of(dpid), mockSwitch);
		getMockSwitchService().setSwitches(switchMap);
		fmc.addService(IFloodlightProviderService.class, mockFloodlightProvider);
		RestApiServer restApi = new RestApiServer();
		fmc.addService(IRestApiService.class, restApi);
		fmc.addService(IOFSwitchService.class, switchService);
		restApi.init(fmc);
		debugCounterService.init(fmc);
		storage.init(fmc);
		staticFlowEntryPusher.init(fmc);
		debugCounterService.init(fmc);
		storage.startUp(fmc);
		createStorageWithFlowEntries();
	}
	@Test
	public void testStaticFlowPush() throws Exception {
		assertEquals(TotalTestRules, staticFlowEntryPusher.countEntries());
		resetToNice(mockSwitch);
		expect(mockSwitch.write(capture(writeCapture))).andReturn(true).anyTimes();
		expect(mockSwitch.write(capture(writeCaptureList))).andReturn(Collections.<OFMessage> emptyList()).anyTimes();
		expect(mockSwitch.getOFFactory()).andReturn(factory).anyTimes();
		expect(mockSwitch.getId()).andReturn(DatapathId.of(dpid)).anyTimes();
		replay(mockSwitch);
		staticFlowEntryPusher.switchAdded(DatapathId.of(dpid));
		verify(mockSwitch);
		assertEquals(true, writeCapture.hasCaptured());
		assertEquals(TotalTestRules, writeCapture.getValues().size());
		OFFlowMod firstFlowMod = (OFFlowMod) writeCapture.getValues().get(2);
		verifyFlowMod(firstFlowMod, FlowMod1);
		OFFlowMod secondFlowMod = (OFFlowMod) writeCapture.getValues().get(1);
		verifyFlowMod(secondFlowMod, FlowMod2);
		OFFlowMod thirdFlowMod = (OFFlowMod) writeCapture.getValues().get(0);
		verifyFlowMod(thirdFlowMod, FlowMod3);
		writeCapture.reset();
		storage.deleteRow(StaticFlowEntryPusher.TABLE_NAME, "TestRule1");
		storage.deleteRow(StaticFlowEntryPusher.TABLE_NAME, "TestRule2");
		assertEquals(1, staticFlowEntryPusher.countEntries());
		assertEquals(2, writeCapture.getValues().size());
		OFFlowMod firstDelete = (OFFlowMod) writeCapture.getValues().get(0);
		FlowMod1 = FlowModUtils.toFlowDeleteStrict(FlowMod1);
		verifyFlowMod(firstDelete, FlowMod1);
		OFFlowMod secondDelete = (OFFlowMod) writeCapture.getValues().get(1);
		FlowMod2 = FlowModUtils.toFlowDeleteStrict(FlowMod2);
		verifyFlowMod(secondDelete, FlowMod2);
		writeCapture.reset();
		FlowMod2 = FlowModUtils.toFlowAdd(FlowMod2);
		FlowMod2 = FlowMod2.createBuilder().setXid(12).build();
		storage.insertRow(StaticFlowEntryPusher.TABLE_NAME, TestRule2);
		assertEquals(2, staticFlowEntryPusher.countEntries());
		assertEquals(1, writeCaptureList.getValues().size());
		List<OFMessage> outList =
				writeCaptureList.getValues().get(0);
		assertEquals(1, outList.size());
		OFFlowMod firstAdd = (OFFlowMod) outList.get(0);
		verifyFlowMod(firstAdd, FlowMod2);
		writeCapture.reset();
		writeCaptureList.reset();
		TestRule3.put(COLUMN_DL_VLAN, 333);
		storage.updateRow(StaticFlowEntryPusher.TABLE_NAME, TestRule3);
		assertEquals(2, staticFlowEntryPusher.countEntries());
		assertEquals(1, writeCaptureList.getValues().size());
		outList = writeCaptureList.getValues().get(0);
		assertEquals(2, outList.size());
		OFFlowMod removeFlowMod = (OFFlowMod) outList.get(0);
		FlowMod3 = FlowModUtils.toFlowDeleteStrict(FlowMod3);
		verifyFlowMod(removeFlowMod, FlowMod3);
		FlowMod3 = FlowModUtils.toFlowAdd(FlowMod3);
		FlowMod3 = FlowMod3.createBuilder().setMatch(MatchUtils.fromString("eth_dst=00:20:30:40:50:60,eth_vlan_vid=333", factory.getVersion())).setXid(14).build();
		OFFlowMod updateFlowMod = (OFFlowMod) outList.get(1);
		verifyFlowMod(updateFlowMod, FlowMod3);
		writeCaptureList.reset();
		storage.updateRow(StaticFlowEntryPusher.TABLE_NAME, TestRule3);
		assertEquals(2, staticFlowEntryPusher.countEntries());
		assertEquals(1, writeCaptureList.getValues().size());
		outList = writeCaptureList.getValues().get(0);
		assertEquals(1, outList.size());
		OFFlowMod modifyFlowMod = (OFFlowMod) outList.get(0);
		FlowMod3 = FlowModUtils.toFlowModifyStrict(FlowMod3);
		List<OFAction> modifiedActions = FlowMod3.getActions();
		FlowMod3 = FlowMod3.createBuilder().setActions(modifiedActions).setXid(19).build();
		verifyFlowMod(modifyFlowMod, FlowMod3);
	}
	IStorageSourceService createStorageWithFlowEntries() {
		return populateStorageWithFlowEntries();
	}
	IStorageSourceService populateStorageWithFlowEntries() {
		Set<String> indexedColumns = new HashSet<String>();
		indexedColumns.add(COLUMN_NAME);
		storage.createTable(StaticFlowEntryPusher.TABLE_NAME, indexedColumns);
		storage.setTablePrimaryKeyName(StaticFlowEntryPusher.TABLE_NAME, COLUMN_NAME);
		storage.insertRow(StaticFlowEntryPusher.TABLE_NAME, TestRule1);
		storage.insertRow(StaticFlowEntryPusher.TABLE_NAME, TestRule2);
		storage.insertRow(StaticFlowEntryPusher.TABLE_NAME, TestRule3);
		return storage;
	}
	@Test
	public void testHARoleChanged() throws IOException {
		assert(staticFlowEntryPusher.entry2dpid.containsValue(TestSwitch1DPID));
		assert(staticFlowEntryPusher.entriesFromStorage.containsValue(FlowMod1));
		assert(staticFlowEntryPusher.entriesFromStorage.containsValue(FlowMod2));
		assert(staticFlowEntryPusher.entriesFromStorage.containsValue(FlowMod3));
        mfp.dispatchRoleChanged(Role.SLAVE);
        assert(staticFlowEntryPusher.entry2dpid.isEmpty());
        assert(staticFlowEntryPusher.entriesFromStorage.isEmpty());
        mfp.dispatchRoleChanged(Role.MASTER);
        assert(staticFlowEntryPusher.entry2dpid.containsValue(TestSwitch1DPID));
        assert(staticFlowEntryPusher.entriesFromStorage.containsValue(FlowMod1));
        assert(staticFlowEntryPusher.entriesFromStorage.containsValue(FlowMod2));
        assert(staticFlowEntryPusher.entriesFromStorage.containsValue(FlowMod3));
	}
}
