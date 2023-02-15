package net.floodlightcontroller.core.internal;
import static org.easymock.EasyMock.createMock;
import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.expectLastCall;
import static org.easymock.EasyMock.replay;
import static org.easymock.EasyMock.reset;
import static org.easymock.EasyMock.verify;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.easymock.Capture;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import net.floodlightcontroller.core.IOFConnectionBackend;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.PortChangeEvent;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.SwitchDescription;
import net.floodlightcontroller.core.SwitchDriverSubHandshakeAlreadyStarted;
import net.floodlightcontroller.core.SwitchDriverSubHandshakeCompleted;
import net.floodlightcontroller.core.SwitchDriverSubHandshakeNotStarted;
import net.floodlightcontroller.core.internal.IOFSwitchManager;
import net.floodlightcontroller.core.internal.OFSwitch;
import net.floodlightcontroller.core.internal.SwitchManagerCounters;
import net.floodlightcontroller.debugcounter.DebugCounterServiceImpl;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFlowAdd;
import org.projectfloodlight.openflow.protocol.OFFlowStatsRequest;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPortConfig;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFPortFeatures;
import org.projectfloodlight.openflow.protocol.OFPortReason;
import org.projectfloodlight.openflow.protocol.OFPortState;
import org.projectfloodlight.openflow.protocol.OFPortStatus;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.OFPort;
public class OFSwitchBaseTest {
    IOFSwitchManager switchManager;
    Map<DatapathId, IOFSwitchBackend> switches;
    private OFMessage testMessage;
    private static class OFSwitchTest extends OFSwitch {
        public OFSwitchTest(IOFConnectionBackend connection, IOFSwitchManager switchManager) {
            super(connection, OFFactories.getFactory(OFVersion.OF_13), switchManager, DatapathId.of(1));
        }
        @Override
        public void setSwitchProperties(SwitchDescription description) {
        }
        @Override
        public OFFactory getOFFactory() {
            return OFFactories.getFactory(OFVersion.OF_13);
        }
        @Override
        public String toString() {
            return "OFSwitchTest";
        }
    }
    private OFSwitchTest sw;
    private OFPortDesc p1a;
    private OFPortDesc p1b;
    private OFPortDesc p2a;
    private OFPortDesc p2b;
    private OFPortDesc p3;
    private final OFPortDesc portFoo1 = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc().setPortNo(OFPort.of(11)).setName("foo").build();
    private final OFPortDesc portFoo2 = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc().setPortNo(OFPort.of(12)).setName("foo").build();
    private final OFPortDesc portBar1 = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc().setPortNo(OFPort.of(11)).setName("bar").build();
    private final OFPortDesc portBar2 = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc().setPortNo(OFPort.of(12)).setName("bar").build();
    private final PortChangeEvent portFoo1Add =
            new PortChangeEvent(portFoo1, PortChangeType.ADD);
    private final PortChangeEvent portFoo2Add =
            new PortChangeEvent(portFoo2, PortChangeType.ADD);
    private final PortChangeEvent portBar1Add =
            new PortChangeEvent(portBar1, PortChangeType.ADD);
    private final PortChangeEvent portBar2Add =
            new PortChangeEvent(portBar2, PortChangeType.ADD);
    private final PortChangeEvent portFoo1Del =
            new PortChangeEvent(portFoo1, PortChangeType.DELETE);
    private final PortChangeEvent portFoo2Del =
            new PortChangeEvent(portFoo2, PortChangeType.DELETE);
    private final PortChangeEvent portBar1Del =
            new PortChangeEvent(portBar1, PortChangeType.DELETE);
    private final PortChangeEvent portBar2Del =
            new PortChangeEvent(portBar2, PortChangeType.DELETE);
    private Capture<Iterable<OFMessage>> capturedMessage;
    private OFFactory factory;
    @Before
    public void setUp() throws Exception {
        IDebugCounterService debugCounter = new DebugCounterServiceImpl();
        switchManager = createMock(IOFSwitchManager.class);
        SwitchManagerCounters counters = new SwitchManagerCounters(debugCounter);
        expect(switchManager.getCounters()).andReturn(counters).anyTimes();
        replay(switchManager);
        factory = OFFactories.getFactory(OFVersion.OF_13);
        testMessage = factory.buildRoleReply()
                .setXid(1)
                .setRole(OFControllerRole.ROLE_MASTER)
                .build();
        IOFConnectionBackend conn = EasyMock.createNiceMock(IOFConnectionBackend.class);
        capturedMessage = new Capture<Iterable<OFMessage>>();
        expect(conn.write(EasyMock.capture(capturedMessage))).andReturn(Collections.<OFMessage>emptyList()).atLeastOnce();
        expect(conn.getOFFactory()).andReturn(factory).anyTimes();
        expect(conn.getAuxId()).andReturn(OFAuxId.MAIN).anyTimes();
        EasyMock.replay(conn);
        IOFConnectionBackend auxConn = EasyMock.createNiceMock(IOFConnectionBackend.class);
        expect(auxConn.getOFFactory()).andReturn(factory).anyTimes();
        expect(auxConn.getAuxId()).andReturn(OFAuxId.of(1)).anyTimes();
        expect(auxConn.write(EasyMock.capture(capturedMessage))).andReturn(Collections.<OFMessage>emptyList()).once();
        EasyMock.replay(auxConn);
        sw = new OFSwitchTest(conn, switchManager);
        sw.registerConnection(auxConn);
        switches = new ConcurrentHashMap<DatapathId, IOFSwitchBackend>();
        switches.put(sw.getId(), sw);
        reset(switchManager);
        setUpPorts();
    }
    public void setUpPorts() {
        OFPortDesc.Builder pdb = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc();
        pdb.setName("port1");
        pdb.setPortNo(OFPort.of(1));
        Set<OFPortState> portState = new HashSet<OFPortState>();
        portState.add(OFPortState.LINK_DOWN);
        pdb.setState(portState);
        p1a = pdb.build();
        assertFalse("Sanity check portEnabled", !p1a.getState().contains(OFPortState.LINK_DOWN));
        pdb = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc();
        pdb.setName("port1");
        pdb.setPortNo(OFPort.of(1));
        Set<OFPortFeatures> portFeatures = new HashSet<OFPortFeatures>();
        portFeatures.add(OFPortFeatures.PF_1GB_FD);
        pdb.setCurr(portFeatures);
        p1b = pdb.build();
        assertTrue("Sanity check portEnabled", !p1b.getState().contains(OFPortState.LIVE));
        pdb = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc();
        portState = new HashSet<OFPortState>();
        Set<OFPortConfig> portConfig = new HashSet<OFPortConfig>();
        portFeatures = new HashSet<OFPortFeatures>();
        pdb.setName("Port2");
        pdb.setPortNo(OFPort.of(2));
        portConfig.add(OFPortConfig.PORT_DOWN);
        pdb.setConfig(portConfig);
        p2a = pdb.build();
        pdb = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc();
        portState = new HashSet<OFPortState>();
        portConfig = new HashSet<OFPortConfig>();
        portFeatures = new HashSet<OFPortFeatures>();
        pdb.setName("Port2");
        pdb.setPortNo(OFPort.of(2));
        portConfig.add(OFPortConfig.PORT_DOWN);
        pdb.setConfig(portConfig);
        portFeatures.add(OFPortFeatures.PF_100MB_HD);
        pdb.setCurr(portFeatures);
        p2b = pdb.build();
        assertFalse("Sanity check portEnabled", p2a.getState().contains(OFPortState.LIVE));
        pdb = OFFactories.getFactory(OFVersion.OF_13).buildPortDesc();
        pdb.setName("porT3");
        pdb.setPortNo(OFPort.of(3));
        p3 = pdb.build();
        assertTrue("Sanity check portEnabled", !p3.getState().contains(OFPortState.LINK_DOWN));
    }
    private static <T> void assertCollectionEqualsNoOrder(Collection<T> expected,
                                         Collection<T> actual) {
        String msg = String.format("expected=%s, actual=%s",
                                   expected.toString(), actual.toString());
        assertEquals(msg, expected.size(), actual.size());
        for(T e: expected) {
            if (!actual.contains(e)) {
                msg = String.format("Expected element %s not found in " +
                        "actual. expected=%s, actual=%s",
                    e, expected, actual);
                fail(msg);
            }
        }
    }
    @Test
    public void testBasicSetPortOperations() {
        Collection<OFPortDesc> oldPorts = Collections.emptyList();
        Collection<OFPortDesc> oldEnabledPorts = Collections.emptyList();
        Collection<OFPort> oldEnabledPortNumbers = Collections.emptyList();
        List<OFPortDesc> ports = new ArrayList<OFPortDesc>();
        Collection<PortChangeEvent> expectedChanges =
                new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> actualChanges = sw.comparePorts(ports);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertEquals(0, sw.getPorts().size());
        assertEquals(0, sw.getEnabledPorts().size());
        assertEquals(0, sw.getEnabledPortNumbers().size());
        actualChanges = sw.setPorts(ports);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertEquals(0, sw.getPorts().size());
        assertEquals(0, sw.getEnabledPorts().size());
        assertEquals(0, sw.getEnabledPortNumbers().size());
        ports.add(p1a);
        ports.add(p2a);
        PortChangeEvent evP1aAdded =
                new PortChangeEvent(p1a, PortChangeType.ADD);
        PortChangeEvent evP2aAdded =
                new PortChangeEvent(p2a, PortChangeType.ADD);
        expectedChanges.clear();
        expectedChanges.add(evP1aAdded);
        expectedChanges.add(evP2aAdded);
        actualChanges = sw.comparePorts(ports);
        assertEquals(0, sw.getPorts().size());
        assertEquals(0, sw.getEnabledPorts().size());
        assertEquals(0, sw.getEnabledPortNumbers().size());
        assertEquals(2, actualChanges.size());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        actualChanges = sw.setPorts(ports);
        assertEquals(2, actualChanges.size());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        assertTrue("enabled ports should be empty",
                   sw.getEnabledPortNumbers().isEmpty());
        assertTrue("enabled ports should be empty",
                   sw.getEnabledPorts().isEmpty());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        assertEquals(null, sw.getPort(OFPort.of(3)));
        assertEquals(null, sw.getPort("port3"));
        oldPorts = sw.getPorts();
        oldEnabledPorts = sw.getEnabledPorts();
        oldEnabledPortNumbers = sw.getEnabledPortNumbers();
        expectedChanges.clear();
        actualChanges = sw.comparePorts(ports);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertEquals(oldPorts, sw.getPorts());
        assertEquals(oldEnabledPorts, sw.getEnabledPorts());
        assertEquals(oldEnabledPortNumbers, sw.getEnabledPortNumbers());
        actualChanges = sw.setPorts(ports);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertEquals(oldPorts, sw.getPorts());
        assertEquals(oldEnabledPorts, sw.getEnabledPorts());
        assertEquals(oldEnabledPortNumbers, sw.getEnabledPortNumbers());
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        assertTrue("enabled ports should be empty",
                   sw.getEnabledPortNumbers().isEmpty());
        assertTrue("enabled ports should be empty",
                   sw.getEnabledPorts().isEmpty());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        assertEquals(null, sw.getPort(OFPort.of(3)));
        assertEquals(null, sw.getPort("port3"));
        oldPorts = sw.getPorts();
        oldEnabledPorts = sw.getEnabledPorts();
        oldEnabledPortNumbers = sw.getEnabledPortNumbers();
        ports.clear();
        ports.add(p2a);
        ports.add(p1b);
        PortChangeEvent evP1bUp = new PortChangeEvent(p1b, PortChangeType.UP);
        actualChanges = sw.comparePorts(ports);
        assertEquals(oldPorts, sw.getPorts());
        assertEquals(oldEnabledPorts, sw.getEnabledPorts());
        assertEquals(oldEnabledPortNumbers, sw.getEnabledPortNumbers());
        assertEquals(1, actualChanges.size());
        assertTrue("No UP event for port1", actualChanges.contains(evP1bUp));
        actualChanges = sw.setPorts(ports);
        assertEquals(1, actualChanges.size());
        assertTrue("No UP event for port1", actualChanges.contains(evP1bUp));
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        List<OFPortDesc> enabledPorts = new ArrayList<OFPortDesc>();
        enabledPorts.add(p1b);
        List<OFPort> enabledPortNumbers = new ArrayList<OFPort>();
        enabledPortNumbers.add(OFPort.of(1));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1b, sw.getPort(OFPort.of(1)));
        assertEquals(p1b, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        assertEquals(null, sw.getPort(OFPort.of(3)));
        assertEquals(null, sw.getPort("port3"));
        oldPorts = sw.getPorts();
        oldEnabledPorts = sw.getEnabledPorts();
        oldEnabledPortNumbers = sw.getEnabledPortNumbers();
        ports.clear();
        ports.add(p2b);
        ports.add(p1b);
        PortChangeEvent evP2bModified =
                new PortChangeEvent(p2b, PortChangeType.OTHER_UPDATE);
        actualChanges = sw.comparePorts(ports);
        assertEquals(oldPorts, sw.getPorts());
        assertEquals(oldEnabledPorts, sw.getEnabledPorts());
        assertEquals(oldEnabledPortNumbers, sw.getEnabledPortNumbers());
        assertEquals(1, actualChanges.size());
        assertTrue("No OTHER_CHANGE event for port2",
                   actualChanges.contains(evP2bModified));
        actualChanges = sw.setPorts(ports);
        assertEquals(1, actualChanges.size());
        assertTrue("No OTHER_CHANGE event for port2",
                   actualChanges.contains(evP2bModified));
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts = new ArrayList<OFPortDesc>();
        enabledPorts.add(p1b);
        enabledPortNumbers = new ArrayList<OFPort>();
        enabledPortNumbers.add(OFPort.of(1));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1b, sw.getPort(OFPort.of(1)));
        assertEquals(p1b, sw.getPort("port1"));
        assertEquals(p2b, sw.getPort(OFPort.of(2)));
        assertEquals(p2b, sw.getPort("port2"));
        assertEquals(null, sw.getPort(OFPort.of(3)));
        assertEquals(null, sw.getPort("port3"));
        oldPorts = sw.getPorts();
        oldEnabledPorts = sw.getEnabledPorts();
        oldEnabledPortNumbers = sw.getEnabledPortNumbers();
        ports.clear();
        ports.add(p2a);
        ports.add(p1a);
        ports.add(p3);
        PortChangeEvent evP1aDown =
                new PortChangeEvent(p1a, PortChangeType.DOWN);
        PortChangeEvent evP2aModified =
                new PortChangeEvent(p2a, PortChangeType.OTHER_UPDATE);
        PortChangeEvent evP3Add =
                new PortChangeEvent(p3, PortChangeType.ADD);
        expectedChanges.clear();
        expectedChanges.add(evP1aDown);
        expectedChanges.add(evP2aModified);
        expectedChanges.add(evP3Add);
        actualChanges = sw.comparePorts(ports);
        assertEquals(oldPorts, sw.getPorts());
        assertEquals(oldEnabledPorts, sw.getEnabledPorts());
        assertEquals(oldEnabledPortNumbers, sw.getEnabledPortNumbers());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        actualChanges = sw.setPorts(ports);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPorts.add(p3);
        enabledPortNumbers.clear();
        enabledPortNumbers.add(OFPort.of(3));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        assertEquals(p3, sw.getPort(OFPort.of(3)));
        assertEquals(p3, sw.getPort("port3"));
        oldPorts = sw.getPorts();
        oldEnabledPorts = sw.getEnabledPorts();
        oldEnabledPortNumbers = sw.getEnabledPortNumbers();
        ports.clear();
        ports.add(p3);
        PortChangeEvent evP1aDel =
                new PortChangeEvent(p1a, PortChangeType.DELETE);
        PortChangeEvent evP2aDel =
                new PortChangeEvent(p2a, PortChangeType.DELETE);
        expectedChanges.clear();
        expectedChanges.add(evP1aDel);
        expectedChanges.add(evP2aDel);
        actualChanges = sw.comparePorts(ports);
        assertEquals(oldPorts, sw.getPorts());
        assertEquals(oldEnabledPorts, sw.getEnabledPorts());
        assertEquals(oldEnabledPortNumbers, sw.getEnabledPortNumbers());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        actualChanges = sw.setPorts(ports);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPorts.add(p3);
        enabledPortNumbers.clear();
        enabledPortNumbers.add(OFPort.of(3));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p3, sw.getPort(OFPort.of(3)));
        assertEquals(p3, sw.getPort("port3"));
    }
    @Test
    public void testBasicPortStatusOperation() {
        OFPortStatus.Builder builder = sw.getOFFactory().buildPortStatus();
        List<OFPortDesc> ports = new ArrayList<OFPortDesc>();
        ports.add(p1a);
        ports.add(p2a);
        PortChangeEvent evP1aAdded =
                new PortChangeEvent(p1a, PortChangeType.ADD);
        PortChangeEvent evP2aAdded =
                new PortChangeEvent(p2a, PortChangeType.ADD);
        Collection<PortChangeEvent> expectedChanges =
                new ArrayList<PortChangeEvent>();
        expectedChanges.add(evP1aAdded);
        expectedChanges.add(evP2aAdded);
        Collection<PortChangeEvent> actualChanges = sw.comparePorts(ports);
        assertEquals(0, sw.getPorts().size());
        assertEquals(0, sw.getEnabledPorts().size());
        assertEquals(0, sw.getEnabledPortNumbers().size());
        assertEquals(2, actualChanges.size());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        actualChanges = sw.setPorts(ports);
        assertEquals(2, actualChanges.size());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        assertTrue("enabled ports should be empty",
                   sw.getEnabledPortNumbers().isEmpty());
        assertTrue("enabled ports should be empty",
                   sw.getEnabledPorts().isEmpty());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        ports.clear();
        ports.add(p2a);
        ports.add(p1b);
        builder.setReason(OFPortReason.MODIFY);
        builder.setDesc(p1b);
        PortChangeEvent evP1bUp = new PortChangeEvent(p1b, PortChangeType.UP);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        expectedChanges.add(evP1bUp);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        List<OFPortDesc> enabledPorts = new ArrayList<OFPortDesc>();
        enabledPorts.add(p1b);
        List<OFPort> enabledPortNumbers = new ArrayList<OFPort>();
        enabledPortNumbers.add(OFPort.of(1));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1b, sw.getPort(OFPort.of(1)));
        assertEquals(p1b, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        ports.clear();
        ports.add(p2b);
        ports.add(p1b);
        PortChangeEvent evP2bModified =
                new PortChangeEvent(p2b, PortChangeType.OTHER_UPDATE);
        builder.setReason(OFPortReason.MODIFY);
        builder.setDesc(p2b);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        expectedChanges.add(evP2bModified);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts = new ArrayList<OFPortDesc>();
        enabledPorts.add(p1b);
        enabledPortNumbers = new ArrayList<OFPort>();
        enabledPortNumbers.add(OFPort.of(1));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1b, sw.getPort(OFPort.of(1)));
        assertEquals(p1b, sw.getPort("port1"));
        assertEquals(p2b, sw.getPort(OFPort.of(2)));
        assertEquals(p2b, sw.getPort("port2"));
        assertEquals(null, sw.getPort(OFPort.of(3)));
        assertEquals(null, sw.getPort("port3"));
        ports.clear();
        ports.add(p2b);
        ports.add(p1a);
        builder.setReason(OFPortReason.ADD);
        builder.setDesc(p1a);
        PortChangeEvent evP1aDown =
                new PortChangeEvent(p1a, PortChangeType.DOWN);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        expectedChanges.add(evP1aDown);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPortNumbers.clear();
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(p2b, sw.getPort(OFPort.of(2)));
        assertEquals(p2b, sw.getPort("port2"));
        ports.clear();
        ports.add(p2a);
        ports.add(p1a);
        builder.setReason(OFPortReason.ADD);
        builder.setDesc(p2a);
        PortChangeEvent evP2aModify =
                new PortChangeEvent(p2a, PortChangeType.OTHER_UPDATE);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        expectedChanges.add(evP2aModify);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPortNumbers.clear();
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(p2a, sw.getPort(OFPort.of(2)));
        assertEquals(p2a, sw.getPort("port2"));
        ports.clear();
        ports.add(p1a);
        builder.setReason(OFPortReason.DELETE);
        builder.setDesc(p2a);
        PortChangeEvent evP2aDel =
                new PortChangeEvent(p2a, PortChangeType.DELETE);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        expectedChanges.add(evP2aDel);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPortNumbers.clear();
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(null, sw.getPort(OFPort.of(2)));
        assertEquals(null, sw.getPort("port2"));
        ports.clear();
        ports.add(p1a);
        builder.setReason(OFPortReason.DELETE);
        builder.setDesc(p2a);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPortNumbers.clear();
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1a, sw.getPort(OFPort.of(1)));
        assertEquals(p1a, sw.getPort("port1"));
        assertEquals(null, sw.getPort(OFPort.of(2)));
        assertEquals(null, sw.getPort("port2"));
        ports.clear();
        builder.setReason(OFPortReason.DELETE);
        builder.setDesc(p1a);
        PortChangeEvent evP1aDel =
                new PortChangeEvent(p1a, PortChangeType.DELETE);
        actualChanges = sw.processOFPortStatus(builder.build());
        expectedChanges.clear();
        expectedChanges.add(evP1aDel);
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPortNumbers.clear();
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(null, sw.getPort(OFPort.of(1)));
        assertEquals(null, sw.getPort("port1"));
        assertEquals(null, sw.getPort(OFPort.of(2)));
        assertEquals(null, sw.getPort("port2"));
        ports.clear();
        ports.add(p3);
        PortChangeEvent evP3Add =
                new PortChangeEvent(p3, PortChangeType.ADD);
        expectedChanges.clear();
        expectedChanges.add(evP3Add);
        builder.setReason(OFPortReason.ADD);
        builder.setDesc(p3);
        actualChanges = sw.processOFPortStatus(builder.build());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPorts.add(p3);
        enabledPortNumbers.clear();
        enabledPortNumbers.add(OFPort.of(3));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(null, sw.getPort(OFPort.of(1)));
        assertEquals(null, sw.getPort("port1"));
        assertEquals(null, sw.getPort(OFPort.of(2)));
        assertEquals(null, sw.getPort("port2"));
        assertEquals(p3, sw.getPort(OFPort.of(3)));
        assertEquals(p3, sw.getPort("port3"));
        ports.clear();
        ports.add(p1b);
        ports.add(p3);
        PortChangeEvent evP1bAdd =
                new PortChangeEvent(p1b, PortChangeType.ADD);
        expectedChanges.clear();
        expectedChanges.add(evP1bAdd);
        builder.setReason(OFPortReason.MODIFY);
        builder.setDesc(p1b);
        actualChanges = sw.processOFPortStatus(builder.build());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPorts.add(p3);
        enabledPorts.add(p1b);
        enabledPortNumbers.clear();
        enabledPortNumbers.add(OFPort.of(3));
        enabledPortNumbers.add(OFPort.of(1));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1b, sw.getPort(OFPort.of(1)));
        assertEquals(p1b, sw.getPort("port1"));
        assertEquals(null, sw.getPort(OFPort.of(2)));
        assertEquals(null, sw.getPort("port2"));
        assertEquals(p3, sw.getPort(OFPort.of(3)));
        assertEquals(p3, sw.getPort("port3"));
        ports.clear();
        ports.add(p1b);
        ports.add(p3);
        expectedChanges.clear();
        builder.setReason(OFPortReason.MODIFY);
        builder.setDesc(p1b);
        actualChanges = sw.processOFPortStatus(builder.build());
        assertCollectionEqualsNoOrder(expectedChanges, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        enabledPorts.clear();
        enabledPorts.add(p3);
        enabledPorts.add(p1b);
        enabledPortNumbers.clear();
        enabledPortNumbers.add(OFPort.of(3));
        enabledPortNumbers.add(OFPort.of(1));
        assertCollectionEqualsNoOrder(enabledPorts, sw.getEnabledPorts());
        assertCollectionEqualsNoOrder(enabledPortNumbers,
                                   sw.getEnabledPortNumbers());
        assertEquals(p1b, sw.getPort(OFPort.of(1)));
        assertEquals(p1b, sw.getPort("port1"));
        assertEquals(null, sw.getPort(OFPort.of(2)));
        assertEquals(null, sw.getPort("port2"));
        assertEquals(p3, sw.getPort(OFPort.of(3)));
        assertEquals(p3, sw.getPort("port3"));
    }
    @Test
    public void testSetPortExceptions() {
        try {
            sw.setPorts(null);
            fail("Expected exception not thrown");
        } catch (NullPointerException e) { };
        List<OFPortDesc> ports = new ArrayList<OFPortDesc>();
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(1)).build());
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(2)).build());
        try {
            sw.setPorts(ports);
            fail("Expected exception not thrown");
        } catch (IllegalArgumentException e) { };
        ports.clear();
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(1)).build());
        ports.add(sw.getOFFactory().buildPortDesc().setName("port2").setPortNo(OFPort.of(1)).build());
        try {
            sw.setPorts(ports);
            fail("Expected exception not thrown");
        } catch (IllegalArgumentException e) { };
        ports.clear();
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(1)).build());
        ports.add(null);
        try {
            sw.setPorts(ports);
            fail("Expected exception not thrown");
        } catch (NullPointerException e) { };
        try {
            sw.getPort((String)null);
            fail("Expected exception not thrown");
        } catch (NullPointerException e) { };
        try {
            sw.comparePorts(null);
            fail("Expected exception not thrown");
        } catch (NullPointerException e) { };
        ports = new ArrayList<OFPortDesc>();
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(1)).build());
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(2)).build());
        try {
            sw.comparePorts(ports);
            fail("Expected exception not thrown");
        } catch (IllegalArgumentException e) { };
        ports.clear();
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(1)).build());
        ports.add(sw.getOFFactory().buildPortDesc().setName("port2").setPortNo(OFPort.of(1)).build());
        try {
            sw.comparePorts(ports);
            fail("Expected exception not thrown");
        } catch (IllegalArgumentException e) { };
        ports.clear();
        ports.add(sw.getOFFactory().buildPortDesc().setName("port1").setPortNo(OFPort.of(1)).build());
        ports.add(null);
        try {
            sw.comparePorts(ports);
            fail("Expected exception not thrown");
        } catch (NullPointerException e) { };
        try {
            sw.getPort((String)null);
            fail("Expected exception not thrown");
        } catch (NullPointerException e) { };
    }
    @Test
    public void testPortStatusExceptions() {
        OFPortStatus.Builder builder = sw.getOFFactory().buildPortStatus();
        try {
            sw.processOFPortStatus(null);
            fail("Expected exception not thrown");
        } catch (NullPointerException e)  { }
        builder.setReason((byte)0x42);
        builder.setDesc(ImmutablePort.create("p1", OFPort.of(1)).toOFPortDesc(sw));
        try {
            sw.processOFPortStatus(builder.build());
            fail("Expected exception not thrown");
        } catch (IllegalArgumentException e)  { }
        builder.setReason(OFPortReason.ADD);
        builder.setDesc(null);
        try {
            sw.processOFPortStatus(builder.build());
            fail("Expected exception not thrown");
        } catch (NullPointerException e)  { }
    }
    private static void assertChangeEvents(Collection<PortChangeEvent> earlyEvents,
                                      Collection<PortChangeEvent> lateEvents,
                                      Collection<PortChangeEvent> anytimeEvents,
                                      Collection<PortChangeEvent> actualEvents) {
        String inputDesc = String.format("earlyEvents=%s, lateEvents=%s, " +
                "anytimeEvents=%s, actualEvents=%s",
                earlyEvents.toString(), lateEvents.toString(), anytimeEvents.toString(), actualEvents.toString());
        Collection<PortChangeEvent> early =
                new ArrayList<PortChangeEvent>(earlyEvents);
        Collection<PortChangeEvent> late =
                new ArrayList<PortChangeEvent>(lateEvents);
        Collection<PortChangeEvent> any =
                new ArrayList<PortChangeEvent>(anytimeEvents);
        for (PortChangeEvent ev: early) {
            assertFalse("Test setup error. Early and late overlap",
                        late.contains(ev));
            assertFalse("Test setup error. Early and anytime overlap",
                        any.contains(ev));
        }
        for (PortChangeEvent ev: late) {
            assertFalse("Test setup error. Late and early overlap",
                        early.contains(ev));
            assertFalse("Test setup error. Late and any overlap",
                        any.contains(ev));
        }
        for (PortChangeEvent ev: any) {
            assertFalse("Test setup error. Anytime and early overlap",
                        early.contains(ev));
            assertFalse("Test setup error. Anytime and late overlap",
                        late.contains(ev));
        }
        for (PortChangeEvent a: actualEvents) {
            if (early.remove(a)) {
                continue;
            }
            if (any.remove(a)) {
                continue;
            }
            if (late.remove(a)) {
                if (!early.isEmpty()) {
                    fail(a + " is in late list, but haven't seen all required " +
                         "early events. " + inputDesc);
                } else {
                    continue;
                }
            }
            fail(a + " was not expected. " + inputDesc);
        }
        if (!early.isEmpty())
            fail("Elements left in early: " + early + ". " + inputDesc);
        if (!late.isEmpty())
            fail("Elements left in late: " + late + ". " + inputDesc);
        if (!any.isEmpty())
            fail("Elements left in any: " + any + ". " + inputDesc);
    }
    @Test
    public void testSetPortNameNumberMappingChange() {
        List<OFPortDesc> ports = new ArrayList<OFPortDesc>();
        Collection<PortChangeEvent> early = new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> late = new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> anytime = new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> actualChanges = null;
        ports.add(portFoo1);
        ports.add(p1a);
        sw.setPorts(ports);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo2);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.setPorts(ports);
        early.add(portFoo1Del);
        late.add(portFoo2Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portBar2);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.setPorts(ports);
        early.add(portFoo2Del);
        late.add(portBar2Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo1);
        ports.add(portBar2);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.setPorts(ports);
        anytime.add(portFoo1Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo2);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.setPorts(ports);
        early.add(portFoo1Del);
        early.add(portBar2Del);
        late.add(portFoo2Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo2);
        ports.add(portBar1);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.setPorts(ports);
        anytime.add(portBar1Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        PortChangeEvent p1bUp = new PortChangeEvent(p1b, PortChangeType.UP);
        PortChangeEvent p3Add = new PortChangeEvent(p3, PortChangeType.ADD);
        ports.clear();
        ports.add(portFoo1);
        ports.add(portBar2);
        ports.add(p1b);
        ports.add(p3);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.setPorts(ports);
        early.add(portFoo2Del);
        early.add(portBar1Del);
        late.add(portFoo1Add);
        late.add(portBar2Add);
        anytime.add(p1bUp);
        anytime.add(p3Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
    }
    @Test
    public void testPortStatusNameNumberMappingChange() {
        List<OFPortDesc> ports = new ArrayList<OFPortDesc>();
        Collection<PortChangeEvent> early = new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> late = new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> anytime = new ArrayList<PortChangeEvent>();
        Collection<PortChangeEvent> actualChanges = null;
        ports.add(portFoo1);
        ports.add(p1a);
        sw.setPorts(ports);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        OFPortStatus.Builder builder = sw.getOFFactory().buildPortStatus();
        builder.setReason(OFPortReason.MODIFY);
        builder.setDesc(portFoo2);
        ports.clear();
        ports.add(portFoo2);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.processOFPortStatus(builder.build());
        early.add(portFoo1Del);
        late.add(portFoo2Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        builder.setReason(OFPortReason.ADD);
        builder.setDesc(portBar2);
        ports.clear();
        ports.add(portBar2);
        ports.add(p1a);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.processOFPortStatus(builder.build());
        early.add(portFoo2Del);
        late.add(portBar2Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo1);
        ports.add(portBar2);
        sw.setPorts(ports);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        builder.setReason(OFPortReason.MODIFY);
        builder.setDesc(portFoo2);
        ports.clear();
        ports.add(portFoo2);
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.processOFPortStatus(builder.build());
        early.add(portFoo1Del);
        early.add(portBar2Del);
        late.add(portFoo2Add);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        builder.setReason(OFPortReason.DELETE);
        builder.setDesc(portFoo1);
        ports.clear();
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.processOFPortStatus(builder.build());
        anytime.add(portFoo2Del);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo1);
        sw.setPorts(ports);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        builder.setReason(OFPortReason.DELETE);
        builder.setDesc(portBar1);
        ports.clear();
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.processOFPortStatus(builder.build());
        anytime.add(portFoo1Del);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        ports.clear();
        ports.add(portFoo1);
        ports.add(portBar2);
        sw.setPorts(ports);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
        builder.setReason(OFPortReason.DELETE);
        builder.setDesc(portFoo2);
        ports.clear();
        early.clear();
        late.clear();
        anytime.clear();
        actualChanges = sw.processOFPortStatus(builder.build());
        anytime.add(portFoo1Del);
        anytime.add(portBar2Del);
        assertChangeEvents(early, late, anytime, actualChanges);
        assertCollectionEqualsNoOrder(ports, sw.getPorts());
    }
    @Test
    public void testSubHandshake() {
        OFMessage m = sw.getOFFactory().buildRoleReply()
                .setXid(1)
                .setRole(OFControllerRole.ROLE_MASTER)
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
    @Test
    public void testMissingConnection() {
       assertFalse("Switch should not have a connection with auxId 5", sw.getConnections().contains(OFAuxId.of(5)));
       try{
           sw.getConnection(OFAuxId.of(5));
           fail("Expected exception not thrown");
       }
    }
    @Test
    public void testInvalidLogicalOFMessageCategory() {
        LogicalOFMessageCategory bad = new LogicalOFMessageCategory("bad", 2);
        assertFalse("Controller should not any logical OFMessage categories", switchManager.isCategoryRegistered(bad));
        reset(switchManager);
        expect(switchManager.isCategoryRegistered(bad)).andReturn(false);
        replay(switchManager);
       try{
           sw.write(testMessage, bad);
           fail("Expected exception not thrown");
       }
       verify(switchManager);
    }
    @Test
    public void testValidLogicalOFMessageCategory() {
        LogicalOFMessageCategory category = new LogicalOFMessageCategory("test", 1);
        assertFalse("Controller should not have any logical OFMessage categories", switchManager.isCategoryRegistered(category));
        reset(switchManager);
        expect(switchManager.isCategoryRegistered(category)).andReturn(true);
        switchManager.handleOutgoingMessage(sw, testMessage);
        expectLastCall().once();
        replay(switchManager);
        sw.write(testMessage, category);
        verify(switchManager);
    }
	@Test
	public void testMasterSlaveWrites() {
		OFFactory factory = OFFactories.getFactory(OFVersion.OF_13);
		OFFlowAdd fa = factory.buildFlowAdd().build();
		OFFlowStatsRequest fsr = factory.buildFlowStatsRequest().build();
		List<OFMessage> msgList = new ArrayList<OFMessage>();
		msgList.add(fa);
		msgList.add(fsr);
		reset(switchManager);
        expect(switchManager.isCategoryRegistered(LogicalOFMessageCategory.MAIN)).andReturn(true).times(6);
        switchManager.handleOutgoingMessage(sw, fa);
        expectLastCall().times(2);
        switchManager.handleOutgoingMessage(sw, fsr);
        expectLastCall().times(4);
        replay(switchManager);
		sw.setControllerRole(OFControllerRole.ROLE_MASTER);
		assertTrue(sw.write(fa));
		assertTrue(sw.write(fsr));
		assertEquals(Collections.<OFMessage>emptyList(), sw.write(msgList));
		sw.setControllerRole(OFControllerRole.ROLE_SLAVE);
	}
}