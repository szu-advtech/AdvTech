package net.floodlightcontroller.packet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Arrays;
import org.junit.Test;
import org.projectfloodlight.openflow.types.EthType;
public class EthernetTest {
    @Test
    public void testToMACAddress() {
        byte[] address = new byte[] { 0x0, 0x11, 0x22, (byte) 0xff,
                (byte) 0xee, (byte) 0xdd};
        assertTrue(Arrays.equals(address, Ethernet
                .toMACAddress("00:11:22:ff:ee:dd")));
        assertTrue(Arrays.equals(address, Ethernet
                .toMACAddress("00:11:22:FF:EE:DD")));
    }
    @Test
    public void testSerialize() {
        Ethernet ethernet = new Ethernet()
            .setDestinationMACAddress("de:ad:be:ef:de:ad")
            .setSourceMACAddress("be:ef:de:ad:be:ef")
            .setEtherType(EthType.of(0));
        byte[] expected = new byte[] { (byte) 0xde, (byte) 0xad, (byte) 0xbe,
                (byte) 0xef, (byte) 0xde, (byte) 0xad, (byte) 0xbe,
                (byte) 0xef, (byte) 0xde, (byte) 0xad, (byte) 0xbe,
                (byte) 0xef, 0x0, 0x0 };
        assertTrue(Arrays.equals(expected, ethernet.serialize()));
    }
    @Test
    public void testToLong() {
        assertEquals(
                281474976710655L,
                Ethernet.toLong(new byte[] { (byte) 0xff, (byte) 0xff,
                        (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff }));
        assertEquals(
                1103823438081L,
                Ethernet.toLong(new byte[] { (byte) 0x01, (byte) 0x01,
                        (byte) 0x01, (byte) 0x01, (byte) 0x01, (byte) 0x01 }));
        assertEquals(
                141289400074368L,
                Ethernet.toLong(new byte[] { (byte) 0x80, (byte) 0x80,
                        (byte) 0x80, (byte) 0x80, (byte) 0x80, (byte) 0x80 }));
    }
}
