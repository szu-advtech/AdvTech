package net.floodlightcontroller.virtualnetwork;
import java.util.List;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFPacketInReason;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.test.MockSyncService;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IListener.Command;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.test.MockThreadPoolService;
import net.floodlightcontroller.core.test.PacketFactory;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.debugevent.MockDebugEventService;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.IEntityClassifierService;
import net.floodlightcontroller.devicemanager.internal.DefaultEntityClassifier;
import net.floodlightcontroller.devicemanager.test.MockDeviceManager;
import net.floodlightcontroller.packet.Data;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPacket;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.UDP;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.restserver.RestApiServer;
import net.floodlightcontroller.test.FloodlightTestCase;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import net.floodlightcontroller.topology.ITopologyService;
import net.floodlightcontroller.virtualnetwork.VirtualNetworkFilter;
public class VirtualNetworkFilterTest extends FloodlightTestCase {
    protected VirtualNetworkFilter vns;
    protected MockDeviceManager deviceService;
    protected static String guid1 = "guid1";
    protected static String net1 = "net1";
    protected static String gw1 = "1.1.1.1";
    protected static String guid2 = "guid2";
    protected static String net2 = "net2";
    protected static String guid3 = "guid3";
    protected static String net3 = "net3";
    protected static String gw2 = "2.2.2.2";
    protected static MacAddress mac1 = MacAddress.of("00:11:22:33:44:55");
    protected static MacAddress mac2 = MacAddress.of("00:11:22:33:44:66");
    protected static MacAddress mac3 = MacAddress.of("00:11:22:33:44:77");
    protected static MacAddress mac4 = MacAddress.of("00:11:22:33:44:88");
    protected static String hostPort1 = "port1";
    protected static String hostPort2 = "port2";
    protected static String hostPort3 = "port3";
    protected static String hostPort4 = "port4";
    protected IOFSwitch sw1;
    protected FloodlightContext cntx;
    protected OFPacketIn mac1ToMac2PacketIn;
    protected IPacket mac1ToMac2PacketIntestPacket;
    protected byte[] mac1ToMac2PacketIntestPacketSerialized;
    protected OFPacketIn mac1ToMac4PacketIn;
    protected IPacket mac1ToMac4PacketIntestPacket;
    protected byte[] mac1ToMac4PacketIntestPacketSerialized;
    protected OFPacketIn mac1ToGwPacketIn;
    protected IPacket mac1ToGwPacketIntestPacket;
    protected byte[] mac1ToGwPacketIntestPacketSerialized;
    protected OFPacketIn packetInDHCPDiscoveryRequest;
    private MockSyncService mockSyncService;
    @Override
    @Before
    public void setUp() throws Exception {
        super.setUp();
        mockSyncService = new MockSyncService();
        FloodlightModuleContext fmc = new FloodlightModuleContext();
        RestApiServer restApi = new RestApiServer();
        deviceService = new MockDeviceManager();
        MockThreadPoolService tps = new MockThreadPoolService();
        ITopologyService topology = createMock(ITopologyService.class);
        vns = new VirtualNetworkFilter();
        DefaultEntityClassifier entityClassifier = new DefaultEntityClassifier();
        fmc.addService(IRestApiService.class, restApi);
        fmc.addService(IFloodlightProviderService.class, getMockFloodlightProvider());
        fmc.addService(IDeviceService.class, deviceService);
        fmc.addService(IThreadPoolService.class, tps);
        fmc.addService(IEntityClassifierService.class, entityClassifier);
        fmc.addService(ITopologyService.class, topology);
        fmc.addService(ISyncService.class, mockSyncService);
        fmc.addService(IDebugCounterService.class, new MockDebugCounterService());
        fmc.addService(IDebugEventService.class, new MockDebugEventService());
        tps.init(fmc);
        deviceService.init(fmc);
        restApi.init(fmc);
        getMockFloodlightProvider().init(fmc);
        entityClassifier.init(fmc);
        tps.startUp(fmc);
        vns.init(fmc);
        deviceService.startUp(fmc);
        restApi.startUp(fmc);
        getMockFloodlightProvider().startUp(fmc);
        vns.startUp(fmc);
        entityClassifier.startUp(fmc);
        expect(topology.isAttachmentPointPort(DatapathId.of(0), OFPort.ZERO)).andReturn(anyBoolean()).anyTimes();
        topology.addListener(deviceService);
        expectLastCall().times(1);
        replay(topology);
        sw1 = EasyMock.createNiceMock(IOFSwitch.class);
        expect(sw1.getId()).andReturn(DatapathId.of(1L)).anyTimes();
        expect(sw1.hasAttribute(IOFSwitch.PROP_SUPPORTS_OFPP_TABLE)).andReturn(true).anyTimes();
        expect(sw1.getOFFactory()).andReturn(OFFactories.getFactory(OFVersion.OF_13)).anyTimes();
        replay(sw1);
        mac1ToMac2PacketIntestPacket = new Ethernet()
            .setDestinationMACAddress(mac2.getBytes())
            .setSourceMACAddress(mac1.getBytes())
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
        mac1ToMac2PacketIntestPacketSerialized = mac1ToMac2PacketIntestPacket.serialize();
        mac1ToMac2PacketIn = OFFactories.getFactory(OFVersion.OF_13).buildPacketIn()
                        .setBufferId(OFBufferId.NO_BUFFER)
                        .setData(mac1ToMac2PacketIntestPacketSerialized)
                        .setReason(OFPacketInReason.NO_MATCH)
                        .build();
        mac1ToMac4PacketIntestPacket = new Ethernet()
        .setDestinationMACAddress(mac4.getBytes())
        .setSourceMACAddress(mac1.getBytes())
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
        mac1ToMac4PacketIntestPacketSerialized = mac1ToMac4PacketIntestPacket.serialize(); 
        mac1ToMac4PacketIn = OFFactories.getFactory(OFVersion.OF_13).buildPacketIn()
                    .setBufferId(OFBufferId.NO_BUFFER)
                    .setData(mac1ToMac4PacketIntestPacketSerialized)
                    .setReason(OFPacketInReason.NO_MATCH)
                    .build();
        mac1ToGwPacketIntestPacket = new Ethernet()
        .setDestinationMACAddress("00:11:33:33:44:55")
        .setSourceMACAddress(mac1.getBytes())
        .setEtherType(EthType.IPv4)
        .setPayload(
            new IPv4()
            .setTtl((byte) 128)
            .setSourceAddress("192.168.1.1")
            .setDestinationAddress(gw1)
            .setPayload(new UDP()
                        .setSourcePort((short) 5000)
                        .setDestinationPort((short) 5001)
                        .setPayload(new Data(new byte[] {0x01}))));
        mac1ToGwPacketIntestPacketSerialized = mac1ToGwPacketIntestPacket.serialize();
        mac1ToGwPacketIn =  OFFactories.getFactory(OFVersion.OF_13).buildPacketIn()
                    .setBufferId(OFBufferId.NO_BUFFER)
                    .setData(mac1ToGwPacketIntestPacketSerialized)
                    .setReason(OFPacketInReason.NO_MATCH)
                    .build();
    }
    @Test
    public void testCreateNetwork() {
        vns.createNetwork(guid1, net1, IPv4Address.of(gw1));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid1));
        assertTrue(vns.nameToGuid.get(net1).equals(guid1));
        assertTrue(vns.guidToGateway.get(guid1).equals(IPv4Address.of(gw1)));
        assertTrue(vns.vNetsByGuid.get(guid1).name.equals(net1));
        assertTrue(vns.vNetsByGuid.get(guid1).guid.equals(guid1));
        assertTrue(vns.vNetsByGuid.get(guid1).gateway.equals(gw1));
        assertTrue(vns.vNetsByGuid.get(guid1).portToMac.size()==0);
        vns.createNetwork(guid2, net2, null);
        assertTrue(vns.nameToGuid.get(net2).equals(guid2));
        assertTrue(vns.guidToGateway.get(guid2) == null);
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).size() == 1);
        assertTrue(vns.vNetsByGuid.get(guid2).name.equals(net2));
        assertTrue(vns.vNetsByGuid.get(guid2).guid.equals(guid2));
        assertTrue(vns.vNetsByGuid.get(guid2).gateway == null);
        assertTrue(vns.vNetsByGuid.get(guid2).portToMac.size()==0);
        vns.createNetwork(guid3, net3, IPv4Address.of(gw1));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid1));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid3));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).size() == 2);
        assertTrue(vns.nameToGuid.get(net3).equals(guid3));
        assertTrue(vns.guidToGateway.get(guid3).equals(IPv4Address.of(gw1)));
        assertTrue(vns.vNetsByGuid.get(guid3).name.equals(net3));
        assertTrue(vns.vNetsByGuid.get(guid3).guid.equals(guid3));
        assertTrue(vns.vNetsByGuid.get(guid3).gateway.equals(gw1));
        assertTrue(vns.vNetsByGuid.get(guid3).portToMac.size()==0);
    }
    @Test
    public void testModifyNetwork() {
        testCreateNetwork();
        vns.createNetwork(guid2, net2, IPv4Address.of(gw1));
        assertTrue(vns.nameToGuid.get(net2).equals(guid2));
        assertTrue(vns.guidToGateway.get(guid2).equals(IPv4Address.of(gw1)));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid1));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid2));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid3));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).size() == 3);
        vns.createNetwork(guid2, "newnet2", null);
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid2));
        assertTrue(vns.vNetsByGuid.get(guid2).gateway.equals(gw1));
        assertTrue(vns.nameToGuid.get("newnet2").equals(guid2));
        assertTrue(vns.vNetsByGuid.get(guid2).name.equals("newnet2"));
        assertFalse(vns.nameToGuid.containsKey(net2));
    }
    @Test
    public void testDeleteNetwork() {
        testModifyNetwork();
        vns.deleteNetwork(guid2);
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid1));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).contains(guid3));
        assertTrue(vns.gatewayToGuid.get(IPv4Address.of(gw1)).size() == 2);
        assertFalse(vns.nameToGuid.containsKey(net2));
        assertFalse(vns.guidToGateway.containsKey(net2));
        assertTrue(vns.vNetsByGuid.get(guid2)==null);
    }
    @Test
    public void testAddHost() {
        testModifyNetwork();
        vns.addHost(mac1, guid1, hostPort1);
        assertTrue(vns.macToGuid.get(mac1).equals(guid1));
        assertTrue(vns.portToMac.get(hostPort1).equals(mac1));
        assertTrue(vns.vNetsByGuid.get(guid1).portToMac.containsValue(mac1));
        vns.addHost(mac2, guid1, hostPort2);
        assertTrue(vns.macToGuid.get(mac2).equals(guid1));
        assertTrue(vns.portToMac.get(hostPort2).equals(mac2));
        assertTrue(vns.vNetsByGuid.get(guid1).portToMac.containsValue(mac2));
        vns.addHost(mac3, guid3, hostPort3);
        vns.addHost(mac4, guid3, hostPort4);
        assertTrue(vns.vNetsByGuid.get(guid3).portToMac.containsValue(mac4));
    }
    @Test
    public void testDeleteHost() {
        testAddHost();
        String host1Guid = vns.macToGuid.get(mac1);
        vns.deleteHost(mac1, null);
        assertFalse(vns.macToGuid.containsKey(mac1));
        assertFalse(vns.portToMac.containsKey(hostPort1));
        assertFalse(vns.vNetsByGuid.get(host1Guid).portToMac.containsValue(mac1));
        String host2Guid = vns.macToGuid.get(vns.portToMac.get(hostPort2));
        vns.deleteHost(null, hostPort2);
        assertFalse(vns.macToGuid.containsKey(mac2));
        assertFalse(vns.portToMac.containsKey(hostPort2));
        assertFalse(vns.vNetsByGuid.get(host2Guid).portToMac.containsValue(mac2));
        String host3Guid = vns.macToGuid.get(mac3);
        vns.deleteHost(mac3, hostPort3);
        assertFalse(vns.macToGuid.containsKey(mac3));
        assertFalse(vns.portToMac.containsKey(hostPort3));
        assertFalse(vns.vNetsByGuid.get(host3Guid).portToMac.containsValue(mac3));
    }
    @Test
    public void testForwarding() {
        testAddHost();
        IOFMessageListener listener = getVirtualNetworkListener();
        cntx = new FloodlightContext();
        IFloodlightProviderService.bcStore.put(cntx,
                           IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                               (Ethernet)mac1ToMac2PacketIntestPacket);
        Command ret = listener.receive(sw1, mac1ToMac2PacketIn, cntx);
        assertTrue(ret == Command.CONTINUE);
        cntx = new FloodlightContext();
        IFloodlightProviderService.bcStore.put(cntx,
                           IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                               (Ethernet)mac1ToMac4PacketIntestPacket);
        ret = listener.receive(sw1, mac1ToMac4PacketIn, cntx);
        assertTrue(ret == Command.STOP);
    }
    @Test
    public void testDefaultGateway() {
        testAddHost();
        IOFMessageListener listener = getVirtualNetworkListener();
        cntx = new FloodlightContext();
        IFloodlightProviderService.bcStore.put(cntx,
                           IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                               (Ethernet)mac1ToGwPacketIntestPacket);
        deviceService.learnEntity(((Ethernet)mac1ToGwPacketIntestPacket).getDestinationMACAddress(),
        		VlanVid.ZERO, IPv4Address.of(gw1), IPv6Address.NONE, DatapathId.NONE, OFPort.ZERO);
        Command ret = listener.receive(sw1, mac1ToGwPacketIn, cntx);
        assertTrue(ret == Command.CONTINUE);
    }
    @Test
    public void testDhcp() {
        IOFMessageListener listener = getVirtualNetworkListener();
        Ethernet dhcpPacket = PacketFactory.DhcpDiscoveryRequestEthernet(mac1);
        OFPacketIn dhcpPacketOf = PacketFactory.DhcpDiscoveryRequestOFPacketIn(sw1, mac1);
        cntx = new FloodlightContext();
        IFloodlightProviderService.bcStore.put(cntx,
                           IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                           dhcpPacket);
        Command ret = listener.receive(sw1, dhcpPacketOf, cntx);
        assertTrue(ret == Command.CONTINUE);
    }
    protected IOFMessageListener getVirtualNetworkListener() {
    	List<IOFMessageListener> listeners = mockFloodlightProvider.getListeners().get(OFType.PACKET_IN);
    	return listeners.get(listeners.indexOf(vns));
    }
}
