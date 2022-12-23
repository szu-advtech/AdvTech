package net.floodlightcontroller.firewall;
import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.replay;
import static org.easymock.EasyMock.verify;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
import net.floodlightcontroller.packet.ARP;
import net.floodlightcontroller.packet.Data;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPacket;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.TCP;
import net.floodlightcontroller.packet.UDP;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.restserver.RestApiServer;
import net.floodlightcontroller.routing.IRoutingDecision;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.storage.memory.MemoryStorageSource;
import net.floodlightcontroller.test.FloodlightTestCase;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketInReason;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv4AddressWithMask;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TransportPort;
public class FirewallTest extends FloodlightTestCase {
    protected FloodlightContext cntx;
    protected OFPacketIn packetIn;
    protected IOFSwitch sw;
    protected IPacket tcpPacket;
    protected IPacket broadcastARPPacket;
    protected IPacket ARPReplyPacket;
    protected IPacket broadcastIPPacket;
    protected IPacket tcpPacketReply;
    protected IPacket broadcastMalformedPacket;
    private Firewall firewall;
    private MockDebugCounterService debugCounterService;
    public static String TestSwitch1DPID = "00:00:00:00:00:00:00:01";
    @Override
    @Before
    public void setUp() throws Exception {
        super.setUp();
        cntx = new FloodlightContext();
        mockFloodlightProvider = getMockFloodlightProvider();
        mockSwitchManager = getMockSwitchService();
        debugCounterService = new MockDebugCounterService(); 
        firewall = new Firewall();
        MemoryStorageSource storageService = new MemoryStorageSource();
        RestApiServer restApi = new RestApiServer();
        DatapathId dpid = DatapathId.of(TestSwitch1DPID);
        sw = EasyMock.createNiceMock(IOFSwitch.class);
        expect(sw.getId()).andReturn(dpid).anyTimes();
        expect(sw.getOFFactory()).andReturn(OFFactories.getFactory(OFVersion.OF_13)).anyTimes();
        replay(sw);
        Map<DatapathId, IOFSwitch> switches = new HashMap<DatapathId, IOFSwitch>();
        switches.put(dpid, sw);
        mockSwitchManager.setSwitches(switches);
        FloodlightModuleContext fmc = new FloodlightModuleContext();
        fmc.addService(IFloodlightProviderService.class, mockFloodlightProvider);
        fmc.addService(IDebugCounterService.class, debugCounterService);
        fmc.addService(IOFSwitchService.class, mockSwitchManager);
        fmc.addService(IFirewallService.class, firewall);
        fmc.addService(IStorageSourceService.class, storageService);
        fmc.addService(IRestApiService.class, restApi);
        debugCounterService.init(fmc);
        storageService.init(fmc);
        restApi.init(fmc);
        firewall.init(fmc);
        debugCounterService.startUp(fmc);
        storageService.startUp(fmc);
        firewall.startUp(fmc);
        this.tcpPacket = new Ethernet()
        .setDestinationMACAddress("00:11:22:33:44:55")
        .setSourceMACAddress("00:44:33:22:11:00")
        .setVlanID((short) 42)
        .setEtherType(EthType.IPv4)
        .setPayload(
                new IPv4()
                .setTtl((byte) 128)
                .setSourceAddress("192.168.1.1")
                .setDestinationAddress("192.168.1.2")
                .setPayload(new TCP()
                .setSourcePort((short) 81)
                .setDestinationPort((short) 80)
                .setPayload(new Data(new byte[] {0x01}))));
        this.broadcastARPPacket = new Ethernet()
        .setDestinationMACAddress("FF:FF:FF:FF:FF:FF")
        .setSourceMACAddress("00:44:33:22:11:00")
        .setVlanID((short) 42)
        .setEtherType(EthType.ARP)
        .setPayload(
                new ARP()
                .setHardwareType(ARP.HW_TYPE_ETHERNET)
                .setProtocolType(ARP.PROTO_TYPE_IP)
                .setOpCode(ARP.OP_REQUEST)
                .setHardwareAddressLength((byte)6)
                .setProtocolAddressLength((byte)4)
                .setSenderHardwareAddress(MacAddress.of("00:44:33:22:11:00"))
                .setSenderProtocolAddress(IPv4Address.of("192.168.1.1"))
                .setTargetHardwareAddress(MacAddress.of("00:00:00:00:00:00"))
                .setTargetProtocolAddress(IPv4Address.of("192.168.1.2"))
                .setPayload(new Data(new byte[] {0x01})));
        this.ARPReplyPacket = new Ethernet()
        .setDestinationMACAddress("00:44:33:22:11:00")
        .setSourceMACAddress("00:11:22:33:44:55")
        .setVlanID((short) 42)
        .setEtherType(EthType.ARP)
        .setPayload(
                new ARP()
                .setHardwareType(ARP.HW_TYPE_ETHERNET)
                .setProtocolType(ARP.PROTO_TYPE_IP)
                .setOpCode(ARP.OP_REQUEST)
                .setHardwareAddressLength((byte)6)
                .setProtocolAddressLength((byte)4)
                .setSenderHardwareAddress(MacAddress.of("00:11:22:33:44:55"))
                .setSenderProtocolAddress(IPv4Address.of("192.168.1.2"))
                .setTargetHardwareAddress(MacAddress.of("00:44:33:22:11:00"))
                .setTargetProtocolAddress(IPv4Address.of("192.168.1.1"))
                .setPayload(new Data(new byte[] {0x01})));
        this.broadcastIPPacket = new Ethernet()
        .setDestinationMACAddress("FF:FF:FF:FF:FF:FF")
        .setSourceMACAddress("00:44:33:22:11:00")
        .setVlanID((short) 42)
        .setEtherType(EthType.IPv4)
        .setPayload(
                new IPv4()
                .setTtl((byte) 128)
                .setSourceAddress("192.168.1.1")
                .setDestinationAddress("192.168.1.255")
                .setPayload(new UDP()
                .setSourcePort((short) 5000)
                .setDestinationPort((short) 5001)
                .setPayload(new Data(new byte[] {0x01}))));
        this.broadcastMalformedPacket = new Ethernet()
        .setDestinationMACAddress("FF:FF:FF:FF:FF:FF")
        .setSourceMACAddress("00:44:33:22:11:00")
        .setVlanID((short) 42)
        .setEtherType(EthType.IPv4)
        .setPayload(
                new IPv4()
                .setTtl((byte) 128)
                .setSourceAddress("192.168.1.1")
                .setDestinationAddress("192.168.1.2")
                .setPayload(new UDP()
                .setSourcePort((short) 5000)
                .setDestinationPort((short) 5001)
                .setPayload(new Data(new byte[] {0x01}))));
        this.tcpPacketReply = new Ethernet()
        .setDestinationMACAddress("00:44:33:22:11:00")
        .setSourceMACAddress("00:11:22:33:44:55")
        .setVlanID((short) 42)
        .setEtherType(EthType.IPv4)
        .setPayload(
                new IPv4()
                .setTtl((byte) 128)
                .setSourceAddress("192.168.1.2")
                .setDestinationAddress("192.168.1.1")
                .setPayload(new TCP()
                .setSourcePort((short) 80)
                .setDestinationPort((short) 81)
                .setPayload(new Data(new byte[] {0x02}))));
    }
    protected void setPacketIn(IPacket packet) {
        byte[] serializedPacket = packet.serialize();
        this.packetIn = OFFactories.getFactory(OFVersion.OF_13).buildPacketIn()
                .setBufferId(OFBufferId.NO_BUFFER)
                .setMatch(OFFactories.getFactory(OFVersion.OF_13).buildMatch().setExact(MatchField.IN_PORT, OFPort.of(1)).build())
                .setData(serializedPacket)
                .setReason(OFPacketInReason.NO_MATCH)
                .build();
        IFloodlightProviderService.bcStore.
        put(cntx,
                IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                (Ethernet)packet);
    }
    @Test
    public void testNoRules() throws Exception {
        firewall.enableFirewall(true);
        this.setPacketIn(tcpPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        assertEquals(0, firewall.rules.size());
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(decision.getRoutingAction(), IRoutingDecision.RoutingAction.DROP);
    }
    @Test
    public void testReadRulesFromStorage() throws Exception {
        FirewallRule rule = new FirewallRule();
        rule.in_port = OFPort.of(2);
        rule.dl_src = MacAddress.of("00:00:00:00:00:01");
        rule.dl_dst = MacAddress.of("00:00:00:00:00:02");
        rule.priority = 1;
        rule.action = FirewallRule.FirewallAction.DROP;
        firewall.addRule(rule);
        rule = new FirewallRule();
        rule.in_port = OFPort.of(3);
        rule.dl_src = MacAddress.of("00:00:00:00:00:02");
        rule.dl_dst = MacAddress.of("00:00:00:00:00:01");
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.tp_dst = TransportPort.of(80);
        rule.priority = 2;
        rule.action = FirewallRule.FirewallAction.ALLOW;
        firewall.addRule(rule);
        List<FirewallRule> rules = firewall.readRulesFromStorage();
        FirewallRule r = rules.get(0);
        assertEquals(r.in_port, OFPort.of(2));
        assertEquals(r.priority, 1);
        assertEquals(r.dl_src, MacAddress.of("00:00:00:00:00:01"));
        assertEquals(r.dl_dst, MacAddress.of("00:00:00:00:00:02"));
        assertEquals(r.action, FirewallRule.FirewallAction.DROP);
        r = rules.get(1);
        assertEquals(r.in_port, OFPort.of(3));
        assertEquals(r.priority, 2);
        assertEquals(r.dl_src, MacAddress.of("00:00:00:00:00:02"));
        assertEquals(r.dl_dst, MacAddress.of("00:00:00:00:00:01"));
        assertEquals(r.nw_proto, IpProtocol.TCP);
        assertEquals(r.tp_dst, TransportPort.of(80));
        assertEquals(r.any_nw_proto, false);
        assertEquals(r.action, FirewallRule.FirewallAction.ALLOW);
    }
    @Test
    public void testRuleInsertionIntoStorage() throws Exception {
        FirewallRule rule = new FirewallRule();
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.priority = 1;
        firewall.addRule(rule);
        List<Map<String, Object>> rulesFromStorage = firewall.getStorageRules();
        assertEquals(1, rulesFromStorage.size());
        assertEquals(Integer.parseInt((String)rulesFromStorage.get(0).get("ruleid")), rule.ruleid);
    }
    @Test
    public void testRuleDeletion() throws Exception {
        FirewallRule rule = new FirewallRule();
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.priority = 1;
        firewall.addRule(rule);
        int rid = rule.ruleid;
        List<Map<String, Object>> rulesFromStorage = firewall.getStorageRules();
        assertEquals(1, rulesFromStorage.size());
        assertEquals(Integer.parseInt((String)rulesFromStorage.get(0).get("ruleid")), rid);
        firewall.deleteRule(rid);
        rulesFromStorage = firewall.getStorageRules();
        assertEquals(0, rulesFromStorage.size());
    }
    @Test
    public void testFirewallDisabled() throws Exception {
        FirewallRule rule = new FirewallRule();
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.priority = 1;
        firewall.addRule(rule);
        this.setPacketIn(tcpPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        assertEquals(1, firewall.rules.size());
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertNull(decision);
    }
    @Test
    public void testSimpleAllowRule() throws Exception {
        firewall.enableFirewall(true);
        FirewallRule rule = new FirewallRule();
        rule.dl_type = EthType.IPv4;
        rule.any_dl_type = false;
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.nw_src_prefix_and_mask = IPv4AddressWithMask.of("192.168.1.2/32");
        rule.any_nw_src = false;
        rule.nw_dst_prefix_and_mask = IPv4AddressWithMask.of("192.168.1.0/24");
        rule.any_nw_dst = false;
        rule.priority = 1;
        firewall.addRule(rule);
        this.setPacketIn(tcpPacketReply);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(IRoutingDecision.RoutingAction.FORWARD_OR_FLOOD, decision.getRoutingAction());
        IRoutingDecision.rtStore.remove(cntx, IRoutingDecision.CONTEXT_DECISION);
        this.setPacketIn(tcpPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(IRoutingDecision.RoutingAction.DROP, decision.getRoutingAction());
    }
    @Test
    public void testOverlappingRules() throws Exception {
        firewall.enableFirewall(true);
        FirewallRule rule = new FirewallRule();
        rule.dl_type = EthType.IPv4;
        rule.any_dl_type = false;
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.tp_dst = TransportPort.of(80);
        rule.priority = 1;
        firewall.addRule(rule);
        rule = new FirewallRule();
        rule.action = FirewallRule.FirewallAction.DROP;
        rule.priority = 2;
        firewall.addRule(rule);
        assertEquals(2, firewall.rules.size());
        this.setPacketIn(tcpPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(decision.getRoutingAction(), IRoutingDecision.RoutingAction.FORWARD_OR_FLOOD);
        IRoutingDecision.rtStore.remove(cntx, IRoutingDecision.CONTEXT_DECISION);
        this.setPacketIn(tcpPacketReply);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(decision.getRoutingAction(), IRoutingDecision.RoutingAction.DROP);
    }
    @Test
    public void testARP() throws Exception {
        firewall.enableFirewall(true);
        this.setPacketIn(broadcastARPPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(IRoutingDecision.RoutingAction.MULTICAST, decision.getRoutingAction());
        IRoutingDecision.rtStore.remove(cntx, IRoutingDecision.CONTEXT_DECISION);
        this.setPacketIn(ARPReplyPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(decision.getRoutingAction(), IRoutingDecision.RoutingAction.DROP);
    }
    @Test
    public void testIPBroadcast() throws Exception {
        firewall.enableFirewall(true);
        firewall.setSubnetMask("255.255.255.0");
        this.setPacketIn(broadcastIPPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(IRoutingDecision.RoutingAction.MULTICAST, decision.getRoutingAction());
    }
    @Test
    public void testMalformedIPBroadcast() throws Exception {
        firewall.enableFirewall(true);
        this.setPacketIn(broadcastMalformedPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(decision.getRoutingAction(), IRoutingDecision.RoutingAction.DROP);
    }
    @Test
    public void testLayer2Rule() throws Exception {
        firewall.enableFirewall(true);
        FirewallRule rule = new FirewallRule();
        rule.dl_src = MacAddress.of("00:44:33:22:11:00");
        rule.any_dl_src = false;
        rule.dl_dst = MacAddress.of("00:11:22:33:44:55");
        rule.any_dl_dst = false;
        rule.priority = 1;
        firewall.addRule(rule);
        rule = new FirewallRule();
        rule.nw_proto = IpProtocol.TCP;
        rule.any_nw_proto = false;
        rule.priority = 2;
        rule.action = FirewallRule.FirewallAction.DROP;
        firewall.addRule(rule);
        this.setPacketIn(tcpPacket);
        firewall.receive(sw, this.packetIn, cntx);
        verify(sw);
        IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
        assertEquals(decision.getRoutingAction(), IRoutingDecision.RoutingAction.FORWARD_OR_FLOOD);
    }
}
