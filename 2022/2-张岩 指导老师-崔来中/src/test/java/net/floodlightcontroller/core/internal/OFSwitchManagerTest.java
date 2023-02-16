package net.floodlightcontroller.core.internal;
import static org.easymock.EasyMock.createMock;
import static org.easymock.EasyMock.createNiceMock;
import static org.easymock.EasyMock.createStrictMock;
import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.expectLastCall;
import static org.easymock.EasyMock.replay;
import static org.easymock.EasyMock.reset;
import static org.easymock.EasyMock.verify;
import static org.easymock.EasyMock.anyObject;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.util.List;
import io.netty.util.Timer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitch.SwitchStatus;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.IOFSwitchDriver;
import net.floodlightcontroller.core.IOFSwitchListener;
import net.floodlightcontroller.core.IShutdownListener;
import net.floodlightcontroller.core.IShutdownService;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.SwitchDescription;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
import net.floodlightcontroller.debugevent.DebugEventService;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.restserver.RestApiServer;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.storage.memory.MemoryStorageSource;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import net.floodlightcontroller.threadpool.ThreadPool;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFPortFeatures;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.OFPort;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.test.MockSyncService;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
public class OFSwitchManagerTest{
    private Controller controller;
    private OFSwitchManager switchManager;
    private final OFFactory factory = OFFactories.getFactory(OFVersion.OF_10);
    private static DatapathId DATAPATH_ID_0 = DatapathId.of(0);
    private static DatapathId DATAPATH_ID_1 = DatapathId.of(1);
    @Before
    public void setUp() throws Exception {
        doSetUp(HARole.ACTIVE);
    }
    public void doSetUp(HARole role) throws Exception {
        FloodlightModuleContext fmc = new FloodlightModuleContext();
        FloodlightProvider cm = new FloodlightProvider();
        fmc.addConfigParam(cm, "role", role.toString());
        controller = (Controller)cm.getServiceImpls().get(IFloodlightProviderService.class);
        fmc.addService(IFloodlightProviderService.class, controller);
        MemoryStorageSource memstorage = new MemoryStorageSource();
        fmc.addService(IStorageSourceService.class, memstorage);
        RestApiServer restApi = new RestApiServer();
        fmc.addService(IRestApiService.class, restApi);
        ThreadPool threadPool = new ThreadPool();
        fmc.addService(IThreadPoolService.class, threadPool);
        MockDebugCounterService debugCounterService = new MockDebugCounterService();
        fmc.addService(IDebugCounterService.class, debugCounterService);
        DebugEventService debugEventService = new DebugEventService();
        fmc.addService(IDebugEventService.class, debugEventService);
        switchManager = new OFSwitchManager();
        fmc.addService(IOFSwitchService.class, switchManager);
        MockSyncService syncService = new MockSyncService();
        fmc.addService(ISyncService.class, syncService);
        IShutdownService shutdownService = createMock(IShutdownService.class);
        shutdownService.registerShutdownListener(anyObject(IShutdownListener.class));
        expectLastCall().anyTimes();
        replay(shutdownService);
        fmc.addService(IShutdownService.class, shutdownService);
        verify(shutdownService);
        threadPool.init(fmc);
        syncService.init(fmc);
        switchManager.init(fmc);
        debugCounterService.init(fmc);
        memstorage.init(fmc);
        debugEventService.init(fmc);
        restApi.init(fmc);
        cm.init(fmc);
        syncService.init(fmc);
        switchManager.startUpBase(fmc);
        debugCounterService.startUp(fmc);
        memstorage.startUp(fmc);
        debugEventService.startUp(fmc);
        threadPool.startUp(fmc);
        restApi.startUp(fmc);
        cm.startUp(fmc);
    }
    @After
    public void tearDown(){
    }
    public Controller getController() {
        return controller;
    }
    private static SwitchDescription createSwitchDescription() {
        return new SwitchDescription();
    }
    private OFFeaturesReply createOFFeaturesReply(DatapathId datapathId) {
        OFFeaturesReply fr = factory.buildFeaturesReply()
                .setXid(0)
                .setDatapathId(datapathId)
                .setPorts(ImmutableList.<OFPortDesc>of())
                .build();
        return fr;
    }
    protected void setupSwitchForAddSwitch(IOFSwitch sw, DatapathId datapathId,
            SwitchDescription description, OFFeaturesReply featuresReply) {
        if (description == null) {
            description = createSwitchDescription();
        }
        if (featuresReply == null) {
            featuresReply = createOFFeaturesReply(datapathId);
        }
        List<OFPortDesc> ports = featuresReply.getPorts();
        expect(sw.getOFFactory()).andReturn(OFFactories.getFactory(OFVersion.OF_10)).anyTimes();
        expect(sw.getStatus()).andReturn(SwitchStatus.MASTER).anyTimes();
        expect(sw.getId()).andReturn(datapathId).anyTimes();
        expect(sw.getSwitchDescription()).andReturn(description).anyTimes();
        expect(sw.getBuffers())
                .andReturn(featuresReply.getNBuffers()).anyTimes();
        expect(sw.getNumTables())
                .andReturn(featuresReply.getNTables()).anyTimes();
        expect(sw.getCapabilities())
                .andReturn(featuresReply.getCapabilities()).anyTimes();
        expect(sw.getActions())
                .andReturn(featuresReply.getActions()).anyTimes();
        expect(sw.getPorts())
                .andReturn(ports).anyTimes();
        expect(sw.attributeEquals(IOFSwitch.SWITCH_SUPPORTS_NX_ROLE, true))
                .andReturn(false).anyTimes();
        expect(sw.getInetAddress()).andReturn(null).anyTimes();
    }
    @Test
    public void testNewSwitchActivated() throws Exception {
        IOFSwitchBackend sw = createMock(IOFSwitchBackend.class);
        setupSwitchForAddSwitch(sw, DATAPATH_ID_0, null, null);
        assertNull(switchManager.getSwitch(DATAPATH_ID_0));
        IOFSwitchListener listener = createStrictMock(IOFSwitchListener.class);
        listener.switchAdded(DATAPATH_ID_0);
        expectLastCall().once();
        listener.switchActivated(DATAPATH_ID_0);
        expectLastCall().once();
        replay(listener);
        switchManager.addOFSwitchListener(listener);
        replay(sw);
        switchManager.switchAdded(sw);
        switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
        verify(sw);
        assertEquals(sw, switchManager.getSwitch(DATAPATH_ID_0));
        controller.processUpdateQueueForTesting();
        verify(listener);
    }
    @Test
    public void testNewSwitchActivatedWhileSlave() throws Exception {
        doSetUp(HARole.STANDBY);
        IOFSwitchBackend sw = createMock(IOFSwitchBackend.class);
        IOFSwitchListener listener = createMock(IOFSwitchListener.class);
        switchManager.addOFSwitchListener(listener);
        expect(sw.getId()).andReturn(DATAPATH_ID_0).anyTimes();
        expect(sw.getStatus()).andReturn(SwitchStatus.MASTER).anyTimes();
        sw.disconnect();
        expectLastCall().once();
        expect(sw.getOFFactory()).andReturn(factory).once();
        switchManager.switchAdded(sw);
        switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
        verify(sw);
        controller.processUpdateQueueForTesting();
        verify(listener);
    }
    private IOFSwitchBackend doActivateSwitchInt(DatapathId datapathId,
                                          SwitchDescription description,
                                          OFFeaturesReply featuresReply,
                                          boolean clearFlows)
                                          throws Exception {
        IOFSwitchBackend sw = createMock(IOFSwitchBackend.class);
        if (featuresReply == null) {
            featuresReply = createOFFeaturesReply(datapathId);
        }
        if (description == null) {
            description = createSwitchDescription();
        }
        setupSwitchForAddSwitch(sw, datapathId, description, featuresReply);
        replay(sw);
        switchManager.switchAdded(sw);
        switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
        verify(sw);
        assertEquals(sw, switchManager.getSwitch(datapathId));
        controller.processUpdateQueueForTesting();
        reset(sw);
        return sw;
    }
    private IOFSwitchBackend doActivateNewSwitch(DatapathId dpid,
                                          SwitchDescription description,
                                          OFFeaturesReply featuresReply)
                                          throws Exception {
        return doActivateSwitchInt(dpid, description, featuresReply, true);
    }
    @Test
    public void testNonexistingSwitchDisconnected() throws Exception {
        IOFSwitchBackend sw = createMock(IOFSwitchBackend.class);
        expect(sw.getId()).andReturn(DATAPATH_ID_1).anyTimes();
        IOFSwitchListener listener = createMock(IOFSwitchListener.class);
        switchManager.addOFSwitchListener(listener);
        replay(sw, listener);
        switchManager.switchDisconnected(sw);
        controller.processUpdateQueueForTesting();
        verify(sw, listener);
        assertNull(switchManager.getSwitch(DATAPATH_ID_1));
    }
    @Test
    public void testSwitchDisconnectedOther() throws Exception {
        IOFSwitch origSw = doActivateNewSwitch(DATAPATH_ID_1, null, null);
        IOFSwitchBackend sw = createMock(IOFSwitchBackend.class);
        expect(sw.getId()).andReturn(DATAPATH_ID_1).anyTimes();
        IOFSwitchListener listener = createMock(IOFSwitchListener.class);
        switchManager.addOFSwitchListener(listener);
        replay(sw, listener);
        switchManager.switchDisconnected(sw);
        controller.processUpdateQueueForTesting();
        verify(sw, listener);
        expect(origSw.getStatus()).andReturn(SwitchStatus.MASTER).anyTimes();
        replay(origSw);
        assertSame(origSw, switchManager.getSwitch(DATAPATH_ID_1));
    }
    @Test
    public void testSwitchActivatedWithAlreadyActiveSwitch() throws Exception {
        SwitchDescription oldDescription = new SwitchDescription(
                "", "", "", "", "Ye Olde Switch");
        SwitchDescription newDescription = new SwitchDescription(
                "", "", "", "", "The new Switch");
        OFFeaturesReply featuresReply = createOFFeaturesReply(DATAPATH_ID_0);
        IOFSwitchBackend oldsw = createMock(IOFSwitchBackend.class);
        setupSwitchForAddSwitch(oldsw, DATAPATH_ID_0, oldDescription, featuresReply);
        replay(oldsw);
        switchManager.switchAdded(oldsw);
        switchManager.switchStatusChanged(oldsw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
        verify(oldsw);
        controller.processUpdateQueueForTesting();
        assertEquals(oldsw, switchManager.getSwitch(DATAPATH_ID_0));
        reset(oldsw);
        expect(oldsw.getId()).andReturn(DATAPATH_ID_0).anyTimes();
        oldsw.cancelAllPendingRequests();
        expectLastCall().once();
        oldsw.disconnect();
        expectLastCall().once();
        IOFSwitchBackend newsw = createMock(IOFSwitchBackend.class);
        setupSwitchForAddSwitch(newsw, DATAPATH_ID_0, newDescription, featuresReply);
        IOFSwitchListener listener = createStrictMock(IOFSwitchListener.class);
        listener.switchRemoved(DATAPATH_ID_0);
        listener.switchAdded(DATAPATH_ID_0);
        listener.switchActivated(DATAPATH_ID_0);
        replay(listener);
        switchManager.addOFSwitchListener(listener);
        replay(newsw, oldsw);
        switchManager.switchAdded(newsw);
        switchManager.switchStatusChanged(newsw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
        verify(newsw, oldsw);
        assertEquals(newsw, switchManager.getSwitch(DATAPATH_ID_0));
        controller.processUpdateQueueForTesting();
        verify(listener);
    }
   @Test
   public void testRemoveActiveSwitch() {
       IOFSwitchBackend sw = createNiceMock(IOFSwitchBackend.class);
       setupSwitchForAddSwitch(sw, DATAPATH_ID_1, null, null);
       replay(sw);
       switchManager.switchAdded(sw);
       switchManager.switchStatusChanged(sw, SwitchStatus.HANDSHAKE, SwitchStatus.MASTER);
       assertEquals(sw, switchManager.getSwitch(DATAPATH_ID_1));
       try {
           switchManager.getAllSwitchMap().remove(DATAPATH_ID_1);
           fail("Expected: UnsupportedOperationException");
       } catch(UnsupportedOperationException e) {
       }
       controller.processUpdateQueueForTesting();
   }
   @Test
   public void testGetActiveSwitch() {
       MockOFConnection connection = new MockOFConnection(DATAPATH_ID_1, OFAuxId.MAIN);
       IOFSwitchBackend sw = new MockOFSwitchImpl(connection);
       sw.setStatus(SwitchStatus.HANDSHAKE);
       assertNull(switchManager.getActiveSwitch(DATAPATH_ID_1));
       switchManager.switchAdded(sw);
       assertNull(switchManager.getActiveSwitch(DATAPATH_ID_1));
       sw.setStatus(SwitchStatus.MASTER);
       assertEquals(sw, switchManager.getActiveSwitch(DATAPATH_ID_1));
       sw.setStatus(SwitchStatus.QUARANTINED);
       assertNull(switchManager.getActiveSwitch(DATAPATH_ID_1));
       sw.setStatus(SwitchStatus.SLAVE);
       assertEquals(sw, switchManager.getActiveSwitch(DATAPATH_ID_1));
       sw.setStatus(SwitchStatus.DISCONNECTED);
       assertNull(switchManager.getActiveSwitch(DATAPATH_ID_1));
       controller.processUpdateQueueForTesting();
   }
   @Test
   public void testNotifySwitchPortChanged() throws Exception {
       DatapathId dpid = DatapathId.of(42);
       OFPortDesc p1 = factory.buildPortDesc()
               .setName("Port1")
               .setPortNo(OFPort.of(1))
               .build();
       OFFeaturesReply fr1 = factory.buildFeaturesReply()
               .setXid(0)
               .setDatapathId(dpid)
               .setPorts(ImmutableList.<OFPortDesc>of(p1))
               .build();
       OFPortDesc p2 = factory.buildPortDesc()
               .setName("Port1")
               .setPortNo(OFPort.of(1))
               .setAdvertised(ImmutableSet.<OFPortFeatures>of(OFPortFeatures.PF_100MB_FD))
               .build();
       OFFeaturesReply fr2 = factory.buildFeaturesReply()
               .setXid(0)
               .setDatapathId(dpid)
               .setPorts(ImmutableList.<OFPortDesc>of(p2))
               .build();
       SwitchDescription desc = createSwitchDescription();
       IOFSwitchBackend sw = doActivateNewSwitch(dpid, desc, fr1);
       IOFSwitchListener listener = createMock(IOFSwitchListener.class);
       switchManager.addOFSwitchListener(listener);
       setupSwitchForAddSwitch(sw, dpid, desc, fr2);
       listener.switchPortChanged(dpid, p2,
                                  PortChangeType.OTHER_UPDATE);
       expectLastCall().once();
       replay(listener);
       replay(sw);
       switchManager.notifyPortChanged(sw, p2,
                                    PortChangeType.OTHER_UPDATE);
       controller.processUpdateQueueForTesting();
       verify(listener);
       verify(sw);
   }
    @Test
    public void testSwitchDriverRegistryBindOrder() {
        IOFSwitchDriver driver1 = createMock(IOFSwitchDriver.class);
        IOFSwitchDriver driver2 = createMock(IOFSwitchDriver.class);
        IOFSwitchDriver driver3 = createMock(IOFSwitchDriver.class);
        IOFSwitchBackend returnedSwitch = null;
        IOFSwitchBackend mockSwitch = createMock(IOFSwitchBackend.class);
        switchManager.addOFSwitchDriver("", driver3);
        switchManager.addOFSwitchDriver("test switch", driver1);
        switchManager.addOFSwitchDriver("test", driver2);
        replay(driver1);
        replay(driver2);
        replay(driver3);
        replay(mockSwitch);
        SwitchDescription description = new SwitchDescription(
                "test switch", "version 0.9", "", "", "");
        reset(driver1);
        reset(driver2);
        reset(driver3);
        reset(mockSwitch);
        mockSwitch.setSwitchProperties(description);
        expectLastCall().once();
        OFFactory factory = OFFactories.getFactory(OFVersion.OF_10);
        expect(driver1.getOFSwitchImpl(description, factory)).andReturn(mockSwitch).once();
        replay(driver1);
        replay(driver2);
        replay(driver3);
        replay(mockSwitch);
        returnedSwitch = switchManager.getOFSwitchInstance(new NullConnection(), description, factory, DatapathId.of(1));
        assertSame(mockSwitch, returnedSwitch);
        verify(driver1);
        verify(driver2);
        verify(driver3);
        verify(mockSwitch);
        description = new SwitchDescription(
                "testFooBar", "version 0.9", "", "", "");
        reset(driver1);
        reset(driver2);
        reset(driver3);
        reset(mockSwitch);
        mockSwitch.setSwitchProperties(description);
        expectLastCall().once();
        expect(driver2.getOFSwitchImpl(description, factory)).andReturn(mockSwitch).once();
        replay(driver1);
        replay(driver2);
        replay(driver3);
        replay(mockSwitch);
        returnedSwitch = switchManager.getOFSwitchInstance(new NullConnection(), description,
                OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
        assertSame(mockSwitch, returnedSwitch);
        verify(driver1);
        verify(driver2);
        verify(driver3);
        verify(mockSwitch);
        description = new SwitchDescription(
                "FooBar", "version 0.9", "", "", "");
        reset(driver1);
        reset(driver2);
        reset(driver3);
        reset(mockSwitch);
        mockSwitch.setSwitchProperties(description);
        expectLastCall().once();
        expect(driver3.getOFSwitchImpl(description, factory)).andReturn(mockSwitch).once();
        replay(driver1);
        replay(driver2);
        replay(driver3);
        replay(mockSwitch);
        returnedSwitch = switchManager.getOFSwitchInstance(new NullConnection(), description, factory, DatapathId.of(1));
        assertSame(mockSwitch, returnedSwitch);
        verify(driver1);
        verify(driver2);
        verify(driver3);
        verify(mockSwitch);
    }
    @Test
    public void testSwitchDriverRegistryNoDriver() {
        IOFSwitchDriver driver = createMock(IOFSwitchDriver.class);
        IOFSwitch returnedSwitch = null;
        IOFSwitchBackend mockSwitch = createMock(IOFSwitchBackend.class);
        switchManager.addOFSwitchDriver("test switch", driver);
        replay(driver);
        replay(mockSwitch);
        SwitchDescription desc = new SwitchDescription("test switch", "version 0.9", "", "", "");
        reset(driver);
        reset(mockSwitch);
        mockSwitch.setSwitchProperties(desc);
        expectLastCall().once();
        expect(driver.getOFSwitchImpl(desc, factory)).andReturn(mockSwitch).once();
        replay(driver);
        replay(mockSwitch);
        returnedSwitch = switchManager.getOFSwitchInstance(new NullConnection(), desc, factory, DatapathId.of(1));
        assertSame(mockSwitch, returnedSwitch);
        verify(driver);
        verify(mockSwitch);
        desc = new SwitchDescription("Foo Bar test switch", "version 0.9", "", "", "");
        reset(driver);
        reset(mockSwitch);
        replay(driver);
        replay(mockSwitch);
        returnedSwitch = switchManager.getOFSwitchInstance(new NullConnection(), desc,
                OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
        assertNotNull(returnedSwitch);
        assertTrue("Returned switch should be OFSwitch",
                   returnedSwitch instanceof OFSwitch);
        assertEquals(desc, returnedSwitch.getSwitchDescription());
        verify(driver);
        verify(mockSwitch);
    }
    @Test
    public void testDriverRegistryExceptions() {
        IOFSwitchDriver driver = createMock(IOFSwitchDriver.class);
        IOFSwitchDriver driver2 = createMock(IOFSwitchDriver.class);
        try {
            switchManager.addOFSwitchDriver("foobar", null);
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        try {
            switchManager.addOFSwitchDriver(null, driver);
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        switchManager.addOFSwitchDriver("foobar",  driver);
        try {
            switchManager.addOFSwitchDriver("foobar",  driver);
            fail("Expected IllegalStateException not thrown");
        } catch (IllegalStateException e) {
        }
        try {
            switchManager.addOFSwitchDriver("foobar",  driver2);
            fail("Expected IllegalStateException not thrown");
        } catch (IllegalStateException e) {
        }
        SwitchDescription description = new SwitchDescription(null, "", "", "", "");
        try {
            switchManager.getOFSwitchInstance(null, description,
                    OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        description = new SwitchDescription("", null, "", "", "");
        try {
            switchManager.getOFSwitchInstance(null, description,
                    OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        description = new SwitchDescription("", "", null, "", "");
        try {
            switchManager.getOFSwitchInstance(null, description,
                    OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        description = new SwitchDescription("", "", "", null, "");
        try {
            switchManager.getOFSwitchInstance(null, description,
                    OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        description = new SwitchDescription("", "", "", "", null);
        try {
            switchManager.getOFSwitchInstance(null, description,
                    OFFactories.getFactory(OFVersion.OF_10), DatapathId.of(1));
            fail("Expected NullPointerException not thrown");
        } catch (NullPointerException e) {
        }
        verify(driver, driver2);
    }
    @Test
    public void testRegisterCategory() {
        Timer timer = createMock(Timer.class);
        replay(timer);
        switchManager = new OFSwitchManager();
        switchManager.loadLogicalCategories();
        assertTrue("Connections should be empty", switchManager.getNumRequiredConnections() == 0);
        switchManager = new OFSwitchManager();
        LogicalOFMessageCategory category = new LogicalOFMessageCategory("aux1", 1);
        switchManager.registerLogicalOFMessageCategory(category);
        switchManager.loadLogicalCategories();
        assertTrue("Required connections should be 1", switchManager.getNumRequiredConnections() == 1);
        switchManager = new OFSwitchManager();
        switchManager.registerLogicalOFMessageCategory(new LogicalOFMessageCategory("aux1", 1));
        switchManager.registerLogicalOFMessageCategory(new LogicalOFMessageCategory("aux1-2", 1));
        switchManager.loadLogicalCategories();
        assertTrue("Required connections should be 1", switchManager.getNumRequiredConnections() == 1);
        switchManager = new OFSwitchManager();
        switchManager.registerLogicalOFMessageCategory(new LogicalOFMessageCategory("aux1", 1));
        switchManager.registerLogicalOFMessageCategory(new LogicalOFMessageCategory("aux2", 2));
        switchManager.loadLogicalCategories();
        assertTrue("Required connections should be 2", switchManager.getNumRequiredConnections() == 2);
    }
    @Test
    public void testRegisterCategoryException() {
        switchManager = new OFSwitchManager();
        switchManager.loadLogicalCategories();
        LogicalOFMessageCategory category = new LogicalOFMessageCategory("test", 1);
        try {
            switchManager.registerLogicalOFMessageCategory(category);
            fail("Expected Unsupported Operation Exception not thrown");
        switchManager = new OFSwitchManager();
        LogicalOFMessageCategory bad = new LogicalOFMessageCategory("bad", 2);
        switchManager.registerLogicalOFMessageCategory(bad);
        try{
            switchManager.loadLogicalCategories();
            fail("Expected exception not thrown");
        switchManager = new OFSwitchManager();
        switchManager.registerLogicalOFMessageCategory(category);
        LogicalOFMessageCategory nonContiguous = new LogicalOFMessageCategory("bad", 3);
        switchManager.registerLogicalOFMessageCategory(nonContiguous);
        try{
            switchManager.loadLogicalCategories();
            fail("Expected exception not thrown");
    }
    @Test
    public void testNewConnectionOpened() {
        MockOFConnection connection = new MockOFConnection(DATAPATH_ID_1, OFAuxId.MAIN);
        OFFeaturesReply featuresReply = createOFFeaturesReply(DATAPATH_ID_1);
        assertTrue(switchManager.getSwitchHandshakeHandlers().isEmpty());
        switchManager.connectionOpened(connection, featuresReply);
        assertTrue(switchManager.getSwitchHandshakeHandlers().size() == 1);
        assertTrue(switchManager.getSwitchHandshakeHandlers().get(0).getDpid().equals(DATAPATH_ID_1));
    }
    @Test
    public void testDuplicateConnectionOpened() {
        testNewConnectionOpened();
        MockOFConnection connection = new MockOFConnection(DATAPATH_ID_1, OFAuxId.MAIN);
        OFFeaturesReply featuresReply = createOFFeaturesReply(DATAPATH_ID_1);
        switchManager.connectionOpened(connection, featuresReply);
        assertTrue(switchManager.getSwitchHandshakeHandlers().size() == 1);
        assertTrue(switchManager.getSwitchHandshakeHandlers().get(0).getDpid().equals(DATAPATH_ID_1));
    }
    @Test
    public void testHandshakeDisconnected() {
        testNewConnectionOpened();
        assertTrue(switchManager.getSwitchHandshakeHandlers().size() == 1);
        switchManager.handshakeDisconnected(DATAPATH_ID_0);
        assertTrue(switchManager.getSwitchHandshakeHandlers().size() == 1);
        switchManager.handshakeDisconnected(DATAPATH_ID_1);
        assertTrue(switchManager.getSwitchHandshakeHandlers().size() == 0);
    }
}
