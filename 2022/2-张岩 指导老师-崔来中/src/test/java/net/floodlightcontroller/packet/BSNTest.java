package net.floodlightcontroller.packet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Arrays;
import org.junit.Test;
import org.projectfloodlight.openflow.types.EthType;
public class BSNTest {       
    protected byte[] probePkt = {
    };
    protected Ethernet getProbePacket() {
        return (Ethernet) new Ethernet()
            .setSourceMACAddress("00:00:00:00:00:04")
            .setDestinationMACAddress("00:00:00:00:00:01")
            .setPayload(new BSN(BSN.BSN_TYPE_PROBE)
	            .setPayload(new BSNPROBE()
		            .setSequenceId(3)
		            .setSrcMac(new byte[] {0x00, 0x00, 0x00, 0x00, 0x00, 0x01})
		            .setDstMac(new byte[] {0x00, 0x00, 0x00, 0x00, 0x00, 0x04})
		            .setSrcSwDpid(0x06)
		            .setSrcPortNo(0x01)
		            )
	            );
    }
    @Test
    public void testSerialize() throws Exception {
        Ethernet pkt = getProbePacket();
        byte[] serialized = pkt.serialize();
        assertTrue(Arrays.equals(probePkt, serialized));
    }
    @Test
    public void testDeserialize() throws Exception {
        Ethernet pkt = (Ethernet) new Ethernet().deserialize(probePkt, 0, probePkt.length);
        byte[] pktarr = pkt.serialize();
        assertTrue(Arrays.equals(probePkt, pktarr));
        Ethernet expected = getProbePacket();
        assertEquals(expected, pkt);
    }
}
