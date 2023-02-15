package net.floodlightcontroller.packet;
import org.junit.Test;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;
public class LLDPOrganizationalTLVTest {
    private final byte[] expected = new byte[] {
        (byte) 254, 13,
        0x0, 0x26, (byte)0xe1,
        0xc,
        0x45, 0x78, 0x74, 0x72, 0x61, 0x49, 0x6e, 0x66, 0x6f
    };
    @Test(expected = IllegalArgumentException.class)
    public void testShortOUI() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.setOUI(new byte[2]);
    }
    @Test(expected = IllegalArgumentException.class)
    public void testLongOUI() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.setOUI(new byte[4]);
    }
    @Test(expected = IllegalArgumentException.class)
    public void testLongInfoString() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.setInfoString(new byte[LLDPOrganizationalTLV.MAX_INFOSTRING_LENGTH + 1]);
    }
    @Test
    public void testMaxInfoString() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.setInfoString(new byte[LLDPOrganizationalTLV.MAX_INFOSTRING_LENGTH]);
    }
    @Test
    public void testInfoString() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.setInfoString("ExtraInfo");
        assertThat(tlv.getInfoString(), is("ExtraInfo".getBytes(Charset.forName("UTF-8"))));
    }
    @Test
    public void testSerialize() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.setLength((short) 13);
        tlv.setOUI(new byte[] {0x0, 0x26, (byte) 0xe1});
        tlv.setSubType((byte) 12);
        tlv.setInfoString("ExtraInfo".getBytes(Charset.forName("UTF-8")));
        assertThat(tlv.getType(), is((byte)127));
        assertThat(tlv.getLength(), is((short)13));
        assertThat(tlv.getOUI(), is(new byte[] {0x0, 0x26, (byte) 0xe1}));
        assertThat(tlv.getSubType(), is((byte)12));
        assertThat(tlv.serialize(), is(expected));
    }
    @Test
    public void testDeserialize() {
        LLDPOrganizationalTLV tlv = new LLDPOrganizationalTLV();
        tlv.deserialize(ByteBuffer.wrap(expected));
        assertThat(tlv.getType(), is((byte)127));
        assertThat(tlv.getLength(), is((short)13));
        assertThat(tlv.getOUI(), is(new byte[] {0x0, 0x26, (byte) 0xe1}));
        assertThat(tlv.getSubType(), is((byte)12));
        assertThat(tlv.getInfoString(), is("ExtraInfo".getBytes(Charset.forName("UTF-8"))));
    }
}
