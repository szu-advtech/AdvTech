package net.floodlightcontroller.test;
import org.junit.Before;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.test.MockFloodlightProvider;
import net.floodlightcontroller.core.test.MockSwitchManager;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.packet.Ethernet;
public class FloodlightTestCase {
    protected MockFloodlightProvider mockFloodlightProvider;
    protected MockSwitchManager mockSwitchManager;
    public MockFloodlightProvider getMockFloodlightProvider() {
        return mockFloodlightProvider;
    }
    public MockSwitchManager getMockSwitchService() {
        return mockSwitchManager;
    }
    public void setMockFloodlightProvider(MockFloodlightProvider mockFloodlightProvider) {
        this.mockFloodlightProvider = mockFloodlightProvider;
    }
    public FloodlightContext parseAndAnnotate(OFMessage m) {
        FloodlightContext bc = new FloodlightContext();
        return parseAndAnnotate(bc, m);
    }
    public static FloodlightContext parseAndAnnotate(FloodlightContext bc, OFMessage m) {
        if (OFType.PACKET_IN.equals(m.getType())) {
            OFPacketIn pi = (OFPacketIn)m;
            Ethernet eth = new Ethernet();
            eth.deserialize(pi.getData(), 0, pi.getData().length);
            IFloodlightProviderService.bcStore.put(bc,
                    IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                    eth);
        }
        return bc;
    }
    @Before
    public void setUp() throws Exception {
        mockFloodlightProvider = new MockFloodlightProvider();
        mockSwitchManager = new MockSwitchManager();
    }
    public static OFPortDesc createOFPortDesc(IOFSwitch sw, String name, int number) {
        OFPortDesc portDesc = sw.getOFFactory().buildPortDesc()
                .setHwAddr(MacAddress.NONE)
                .setPortNo(OFPort.of(number))
                .setName(name)
                .build();
        return portDesc;
    }
}
