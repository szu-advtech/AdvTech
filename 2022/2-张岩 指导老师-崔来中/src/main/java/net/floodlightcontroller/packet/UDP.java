package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.TransportPort;
public class UDP extends BasePacket {
    public static Map<TransportPort, Class<? extends IPacket>> decodeMap;
    public static final TransportPort DHCP_CLIENT_PORT = TransportPort.of(68);
    public static final TransportPort DHCP_SERVER_PORT = TransportPort.of(67);
    public static final TransportPort DHCPv6_CLIENT_PORT = TransportPort.of(546);
    public static final TransportPort DHCPv6_SERVER_PORT = TransportPort.of(546);
    static {
        decodeMap = new HashMap<TransportPort, Class<? extends IPacket>>();
        UDP.decodeMap.put(DHCP_CLIENT_PORT, DHCP.class);
        UDP.decodeMap.put(DHCP_SERVER_PORT, DHCP.class);
        UDP.decodeMap.put(DHCPv6_CLIENT_PORT, DHCPv6.class);
        UDP.decodeMap.put(DHCPv6_CLIENT_PORT, DHCPv6.class);
    }
    protected TransportPort sourcePort;
    protected TransportPort destinationPort;
    protected short length;
    protected short checksum;
    public TransportPort getSourcePort() {
        return sourcePort;
    }
    public UDP setSourcePort(TransportPort sourcePort) {
        this.sourcePort = sourcePort;
        return this;
    }
    public UDP setSourcePort(short sourcePort) {
        this.sourcePort = TransportPort.of(sourcePort);
        return this;
    }
    public TransportPort getDestinationPort() {
        return destinationPort;
    }
    public UDP setDestinationPort(TransportPort destinationPort) {
        this.destinationPort = destinationPort;
        return this;
    }
    public UDP setDestinationPort(short destinationPort) {
        this.destinationPort = TransportPort.of(destinationPort);
        return this;
    }
    public short getLength() {
        return length;
    }
    public short getChecksum() {
        return checksum;
    }
    public UDP setChecksum(short checksum) {
        this.checksum = checksum;
        return this;
    }
    @Override
    public void resetChecksum() {
        this.checksum = 0;
        super.resetChecksum();
    }
    public byte[] serialize() {
        byte[] payloadData = null;
        if (payload != null) {
            payload.setParent(this);
            payloadData = payload.serialize();
        }
        this.length = (short) (8 + ((payloadData == null) ? 0
                : payloadData.length));
        byte[] data = new byte[this.length];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.putShort((short)this.destinationPort.getPort());
        bb.putShort(this.length);
        bb.putShort(this.checksum);
        if (payloadData != null)
            bb.put(payloadData);
        if (this.parent != null && this.parent instanceof IPv4)
            ((IPv4)this.parent).setProtocol(IpProtocol.UDP);
        if (this.checksum == 0) {
            bb.rewind();
            int accumulation = 0;
            if (this.parent != null && this.parent instanceof IPv4) {
                IPv4 ipv4 = (IPv4) this.parent;
                accumulation += ((ipv4.getSourceAddress().getInt() >> 16) & 0xffff)
                        + (ipv4.getSourceAddress().getInt() & 0xffff);
                accumulation += ((ipv4.getDestinationAddress().getInt() >> 16) & 0xffff)
                        + (ipv4.getDestinationAddress().getInt() & 0xffff);
                accumulation += ipv4.getProtocol().getIpProtocolNumber() & 0xff;
                accumulation += this.length & 0xffff;
            }
            for (int i = 0; i < this.length / 2; ++i) {
                accumulation += 0xffff & bb.getShort();
            }
            if (this.length % 2 > 0) {
                accumulation += (bb.get() & 0xff) << 8;
            }
            accumulation = ((accumulation >> 16) & 0xffff)
                    + (accumulation & 0xffff);
            this.checksum = (short) (~accumulation & 0xffff);
            bb.putShort(6, this.checksum);
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
        if (!(obj instanceof UDP))
            return false;
        UDP other = (UDP) obj;
        if (checksum != other.checksum)
            return false;
        if (!destinationPort.equals(other.destinationPort))
            return false;
        if (length != other.length)
            return false;
        if (!sourcePort.equals(other.sourcePort))
            return false;
        return true;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length)
            throws PacketParsingException {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        this.length = bb.getShort();
        this.checksum = bb.getShort();
        ByteBuffer bb_spud = bb.slice();
        byte[] maybe_spud_bytes = new byte[SPUD.MAGIC_CONSTANT.length];
        if (bb_spud.remaining() >= SPUD.MAGIC_CONSTANT.length) {
            bb_spud.get(maybe_spud_bytes, 0, SPUD.MAGIC_CONSTANT.length);
        }
        if (UDP.decodeMap.containsKey(this.destinationPort)) {
            try {
                this.payload = UDP.decodeMap.get(this.destinationPort).getConstructor().newInstance();
            } catch (Exception e) {
                throw new RuntimeException("Failure instantiating class", e);
            }
        } else if (UDP.decodeMap.containsKey(this.sourcePort)) {
            try {
                this.payload = UDP.decodeMap.get(this.sourcePort).getConstructor().newInstance();
            } catch (Exception e) {
                throw new RuntimeException("Failure instantiating class", e);
            }
        } else if (Arrays.equals(maybe_spud_bytes, SPUD.MAGIC_CONSTANT)
                && bb.remaining() >= SPUD.HEADER_LENGTH) {
            this.payload = new SPUD();
        } else {
            this.payload = new Data();
        }
        this.payload = payload.deserialize(data, bb.position(), bb.limit()-bb.position());
        this.payload.setParent(this);
        return this;
    }
}
