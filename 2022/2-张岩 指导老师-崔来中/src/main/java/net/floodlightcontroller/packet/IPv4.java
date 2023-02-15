package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.U8;
public class IPv4 extends BasePacket {
    public static Map<IpProtocol, Class<? extends IPacket>> protocolClassMap;
    static {
        protocolClassMap = new HashMap<IpProtocol, Class<? extends IPacket>>();
        protocolClassMap.put(IpProtocol.ICMP, ICMP.class);
        protocolClassMap.put(IpProtocol.TCP, TCP.class);
        protocolClassMap.put(IpProtocol.UDP, UDP.class);
    }
    public static final byte IPV4_FLAGS_MOREFRAG = 0x1;
    public static final byte IPV4_FLAGS_DONTFRAG = 0x2;
    public static final byte IPV4_FLAGS_MASK = 0x7;
    public static final byte IPV4_FLAGS_SHIFT = 13;
    public static final short IPV4_OFFSET_MASK = (1 << IPV4_FLAGS_SHIFT) - 1;
    protected byte version;
    protected byte headerLength;
    protected byte diffServ;
    protected short totalLength;
    protected short identification;
    protected byte flags;
    protected short fragmentOffset;
    protected byte ttl;
    protected IpProtocol protocol;
    protected short checksum;
    protected IPv4Address sourceAddress;
    protected IPv4Address destinationAddress;
    protected byte[] options;
    protected boolean isTruncated;
    protected boolean isFragment;
    public IPv4() {
        super();
        this.version = 4;
        isTruncated = false;
        isFragment = false;
        protocol = IpProtocol.NONE;
        sourceAddress = IPv4Address.NONE;
        destinationAddress = IPv4Address.NONE;
    }
    public byte getVersion() {
        return version;
    }
    public IPv4 setVersion(byte version) {
        this.version = version;
        return this;
    }
    public byte getHeaderLength() {
        return headerLength;
    }
    public byte getDiffServ() {
        return diffServ;
    }
    public IPv4 setDiffServ(byte diffServ) {
        this.diffServ = diffServ;
        return this;
    }
    public short getTotalLength() {
        return totalLength;
    }
    public short getIdentification() {
        return identification;
    }
    public boolean isTruncated() {
        return isTruncated;
    }
    public void setTruncated(boolean isTruncated) {
        this.isTruncated = isTruncated;
    }
    public boolean isFragment() {
        return isFragment;
    }
    public void setFragment(boolean isFrag) {
        this.isFragment = isFrag;
    }
    public IPv4 setIdentification(short identification) {
        this.identification = identification;
        return this;
    }
    public byte getFlags() {
        return flags;
    }
    public IPv4 setFlags(byte flags) {
        this.flags = flags;
        return this;
    }
    public short getFragmentOffset() {
        return fragmentOffset;
    }
    public IPv4 setFragmentOffset(short fragmentOffset) {
        this.fragmentOffset = fragmentOffset;
        return this;
    }
    public byte getTtl() {
        return ttl;
    }
    public IPv4 setTtl(byte ttl) {
        this.ttl = ttl;
        return this;
    }
    public IpProtocol getProtocol() {
        return protocol;
    }
    public IPv4 setProtocol(IpProtocol protocol) {
        this.protocol = protocol;
        return this;
    }
    public short getChecksum() {
        return checksum;
    }
    public IPv4 setChecksum(short checksum) {
        this.checksum = checksum;
        return this;
    }
    @Override
    public void resetChecksum() {
        this.checksum = 0;
        super.resetChecksum();
    }
    public IPv4Address getSourceAddress() {
        return sourceAddress;
    }
    public IPv4 setSourceAddress(IPv4Address sourceAddress) {
        this.sourceAddress = sourceAddress;
        return this;
    }
    public IPv4 setSourceAddress(int sourceAddress) {
        this.sourceAddress = IPv4Address.of(sourceAddress);
        return this;
    }
    public IPv4 setSourceAddress(String sourceAddress) {
        this.sourceAddress = IPv4Address.of(sourceAddress);
        return this;
    }
    public IPv4Address getDestinationAddress() {
        return destinationAddress;
    }
    public IPv4 setDestinationAddress(IPv4Address destinationAddress) {
        this.destinationAddress = destinationAddress;
        return this;
    }
    public IPv4 setDestinationAddress(int destinationAddress) {
        this.destinationAddress = IPv4Address.of(destinationAddress);
        return this;
    }
    public IPv4 setDestinationAddress(String destinationAddress) {
        this.destinationAddress = IPv4Address.of(destinationAddress);
        return this;
    }
    public byte[] getOptions() {
        return options;
    }
    public IPv4 setOptions(byte[] options) {
        if (options != null && (options.length % 4) > 0)
            throw new IllegalArgumentException(
                    "Options length must be a multiple of 4");
        this.options = options;
        return this;
    }
    @Override
    public byte[] serialize() {
        byte[] payloadData = null;
        if (payload != null) {
            payload.setParent(this);
            payloadData = payload.serialize();
        }
        int optionsLength = 0;
        if (this.options != null)
            optionsLength = this.options.length / 4;
        this.headerLength = (byte) (5 + optionsLength);
                : payloadData.length));
        byte[] data = new byte[this.totalLength];
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.put((byte) (((this.version & 0xf) << 4) | (this.headerLength & 0xf)));
        bb.put(this.diffServ);
        bb.putShort(this.totalLength);
        bb.putShort(this.identification);
        bb.putShort((short)(((this.flags & IPV4_FLAGS_MASK) << IPV4_FLAGS_SHIFT)
                | (this.fragmentOffset & IPV4_OFFSET_MASK)));
        bb.put(this.ttl);
        bb.put((byte)this.protocol.getIpProtocolNumber());
        bb.putShort(this.checksum);
        bb.putInt(this.sourceAddress.getInt());
        bb.putInt(this.destinationAddress.getInt());
        if (this.options != null)
            bb.put(this.options);
        if (payloadData != null)
            bb.put(payloadData);
        if (this.checksum == 0) {
            bb.rewind();
            int accumulation = 0;
                accumulation += 0xffff & bb.getShort();
            }
            accumulation = ((accumulation >> 16) & 0xffff)
                    + (accumulation & 0xffff);
            this.checksum = (short) (~accumulation & 0xffff);
            bb.putShort(10, this.checksum);
        }
        return data;
    }
    @Override
    public IPacket deserialize(byte[] data, int offset, int length)
            throws PacketParsingException {
        ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
        short sscratch;
        this.version = bb.get();
        this.headerLength = (byte) (this.version & 0xf);
        this.version = (byte) ((this.version >> 4) & 0xf);
        if (this.version != 4) {
            throw new PacketParsingException(
                    "Invalid version for IPv4 packet: " +
                    this.version);
        }
        this.diffServ = bb.get();
        this.totalLength = bb.getShort();
        this.identification = bb.getShort();
        sscratch = bb.getShort();
        this.flags = (byte) ((sscratch >> IPV4_FLAGS_SHIFT) & IPV4_FLAGS_MASK);
        this.fragmentOffset = (short) (sscratch & IPV4_OFFSET_MASK);
        this.ttl = bb.get();
        this.protocol = IpProtocol.of(U8.f(bb.get()));
        this.checksum = bb.getShort();
        this.sourceAddress = IPv4Address.of(bb.getInt());
        this.destinationAddress = IPv4Address.of(bb.getInt());
        if (this.headerLength > 5) {
            this.options = new byte[optionsLength];
            bb.get(this.options);
        }
        IPacket payload;
        isFragment = ((this.flags & IPV4_FLAGS_DONTFRAG) == 0) &&
                ((this.flags & IPV4_FLAGS_MOREFRAG) != 0 ||
                this.fragmentOffset != 0);
        if (!isFragment && IPv4.protocolClassMap.containsKey(this.protocol)) {
            Class<? extends IPacket> clazz = IPv4.protocolClassMap.get(this.protocol);
            try {
                payload = clazz.newInstance();
            } catch (Exception e) {
                throw new RuntimeException("Error parsing payload for IPv4 packet", e);
            }
        } else {
            if (log.isTraceEnabled() && isFragment) {
                log.trace("IPv4 fragment detected {}->{}, forward using IP header only",
                        this.sourceAddress.toString(),
                        this.destinationAddress.toString());
            }
            payload = new Data();
        }
        int remLength = bb.limit()-bb.position();
        if (remLength < payloadLength)
            payloadLength = bb.limit()-bb.position();
        this.payload = payload.deserialize(data, bb.position(), payloadLength);
        this.payload.setParent(this);
        if (this.totalLength > length)
            this.isTruncated = true;
        else
            this.isTruncated = false;
        return this;
    }
    public static int toIPv4Address(String ipAddress) {
        if (ipAddress == null)
            throw new IllegalArgumentException("Specified IPv4 address must" +
                "contain 4 sets of numerical digits separated by periods");
        String[] octets = ipAddress.split("\\.");
        if (octets.length != 4)
            throw new IllegalArgumentException("Specified IPv4 address must" +
                "contain 4 sets of numerical digits separated by periods");
        int result = 0;
        for (int i = 0; i < 4; ++i) {
            int oct = Integer.valueOf(octets[i]);
            if (oct > 255 || oct < 0)
                throw new IllegalArgumentException("Octet values in specified" +
                        " IPv4 address must be 0 <= value <= 255");
        }
        return result;
    }
    public static int toIPv4Address(byte[] ipAddress) {
        int ip = 0;
        for (int i = 0; i < 4; i++) {
          ip |= t;
        }
        return ip;
    }
    public static String fromIPv4Address(int ipAddress) {
        StringBuffer sb = new StringBuffer();
        int result = 0;
        for (int i = 0; i < 4; ++i) {
            sb.append(Integer.valueOf(result).toString());
            if (i != 3)
                sb.append(".");
        }
        return sb.toString();
    }
    public static String fromIPv4AddressCollection(Collection<Integer> ipAddresses) {
        if (ipAddresses == null)
            return "null";
        StringBuffer sb = new StringBuffer();
        sb.append("[");
        for (Integer ip : ipAddresses) {
            sb.append(fromIPv4Address(ip));
            sb.append(",");
        }
        sb.replace(sb.length()-1, sb.length(), "]");
        return sb.toString();
    }
    public static byte[] toIPv4AddressBytes(String ipAddress) {
        String[] octets = ipAddress.split("\\.");
        if (octets.length != 4)
            throw new IllegalArgumentException("Specified IPv4 address must" +
                "contain 4 sets of numerical digits separated by periods");
        byte[] result = new byte[4];
        for (int i = 0; i < 4; ++i) {
            result[i] = Integer.valueOf(octets[i]).byteValue();
        }
        return result;
    }
    public static byte[] toIPv4AddressBytes(int ipAddress) {
        return new byte[] {
                (byte)(ipAddress >>> 24),
                (byte)(ipAddress >>> 16),
                (byte)(ipAddress >>> 8),
                (byte)ipAddress};
    }
    @Override
    public int hashCode() {
        final int prime = 2521;
        int result = super.hashCode();
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (!(obj instanceof IPv4))
            return false;
        IPv4 other = (IPv4) obj;
        if (checksum != other.checksum)
            return false;
        if (!destinationAddress.equals(other.destinationAddress))
            return false;
        if (diffServ != other.diffServ)
            return false;
        if (flags != other.flags)
            return false;
        if (fragmentOffset != other.fragmentOffset)
            return false;
        if (headerLength != other.headerLength)
            return false;
        if (identification != other.identification)
            return false;
        if (!Arrays.equals(options, other.options))
            return false;
        if (!protocol.equals(other.protocol))
            return false;
        if (!sourceAddress.equals(other.sourceAddress))
            return false;
        if (totalLength != other.totalLength)
            return false;
        if (ttl != other.ttl)
            return false;
        if (version != other.version)
            return false;
        return true;
    }
}
