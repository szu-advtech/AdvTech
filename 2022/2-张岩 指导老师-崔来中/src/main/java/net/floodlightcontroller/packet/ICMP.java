package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import org.projectfloodlight.openflow.types.IpProtocol;
public class ICMP extends BasePacket {
    protected byte icmpType;
    protected byte icmpCode;
    protected short checksum;
    public static final Map<Byte, Short> paddingMap;
    public static final byte ECHO_REPLY = 0x0;
    public static final byte ECHO_REQUEST = 0x8;
    public static final byte TIME_EXCEEDED = 0xB;
    public static final byte DESTINATION_UNREACHABLE = 0x3;
    public static final byte CODE_PORT_UNREACHABLE = 0x3;
    static {
        paddingMap = new HashMap<Byte, Short>();
        ICMP.paddingMap.put(ICMP.ECHO_REPLY, (short) 0);
        ICMP.paddingMap.put(ICMP.ECHO_REQUEST, (short) 0);
        ICMP.paddingMap.put(ICMP.TIME_EXCEEDED, (short) 4);
        ICMP.paddingMap.put(ICMP.DESTINATION_UNREACHABLE, (short) 4);
    }
    public byte getIcmpType() {
        return icmpType;
    }
    public ICMP setIcmpType(byte icmpType) {
        this.icmpType = icmpType;
        return this;
    }
    public byte getIcmpCode() {
        return icmpCode;
    }
    public ICMP setIcmpCode(byte icmpCode) {
        this.icmpCode = icmpCode;
        return this;
    }
    public short getChecksum() {
        return checksum;
    }
    public ICMP setChecksum(short checksum) {
        this.checksum = checksum;
        return this;
    }
    @Override
    public byte[] serialize() {
        short padding = 0;
        if (paddingMap.containsKey(this.icmpType))
            padding = paddingMap.get(this.icmpType);
        int length = 4 + padding;
        byte[] payloadData = null;
        if (payload != null) {
            payload.setParent(this);
            payloadData = payload.serialize();
            length += payloadData.length;
        }
        byte[] data = new byte[length];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.put(this.icmpType);
        bb.put(this.icmpCode);
        bb.putShort(this.checksum);
        for (int i = 0; i < padding; i++)
            bb.put((byte) 0);
        if (payloadData != null)
            bb.put(payloadData);
        if (this.parent != null && this.parent instanceof IPv4)
            ((IPv4)this.parent).setProtocol(IpProtocol.ICMP);
        if (this.checksum == 0) {
            bb.rewind();
            int accumulation = 0;
            for (int i = 0; i < length / 2; ++i) {
                accumulation += 0xffff & bb.getShort();
            }
            if (length % 2 > 0) {
                accumulation += (bb.get() & 0xff) << 8;
            }
            accumulation = ((accumulation >> 16) & 0xffff)
                    + (accumulation & 0xffff);
            this.checksum = (short) (~accumulation & 0xffff);
            bb.putShort(2, this.checksum);
        }
        return data;
    }
    @Override
    public int hashCode() {
        final int prime = 5807;
        int result = super.hashCode();
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (!(obj instanceof ICMP))
            return false;
        ICMP other = (ICMP) obj;
        if (icmpType != other.icmpType)
            return false;
        if (icmpCode != other.icmpCode)
            return false;
        if (checksum != other.checksum)
            return false;
        return true;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length)
            throws PacketParsingException {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        this.icmpType = bb.get();
        this.icmpCode = bb.get();
        this.checksum = bb.getShort();
        short padding = 0;
        if (paddingMap.containsKey(this.icmpType))
            padding = paddingMap.get(this.icmpType);
        bb.position(bb.position() + padding);
        this.payload = new Data();
        this.payload = payload.deserialize(data, bb.position(), bb.limit()-bb.position());
        this.payload.setParent(this);
        return this;
    }
}
