package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.IpProtocol;
public class IPv6 extends BasePacket {
	public static Map<IpProtocol, Class<? extends IPacket>> nextHeaderClassMap;
	static {
		nextHeaderClassMap = new HashMap<IpProtocol, Class<? extends IPacket>>();
		nextHeaderClassMap.put(IpProtocol.TCP, TCP.class);
		nextHeaderClassMap.put(IpProtocol.UDP, UDP.class);
		nextHeaderClassMap.put(IpProtocol.IPv6_ICMP, ICMPv6.class);
		nextHeaderClassMap.put(IpProtocol.of((short)150), DHCPv6.class);
	}
	public static final int HEADER_LENGTH = 40;
	protected byte version;
	protected byte trafficClass;
	protected int flowLabel;
	protected short payloadLength;
	protected IpProtocol nextHeader;
	protected byte hopLimit;
	protected IPv6Address sourceAddress;
	protected IPv6Address destinationAddress;
	public IPv6() {
		super();
		this.version = 6;
		nextHeader = IpProtocol.NONE;
		sourceAddress = IPv6Address.NONE;
		destinationAddress = IPv6Address.NONE;
	}
	public byte getVersion() {
		return version;
	}
	public IPv6 setVersion(byte version) {
		this.version = version;
		return this;
	}
	public byte getTrafficClass() {
		return trafficClass;
	}
	public IPv6 setTrafficClass(byte trafficClass) {
		this.trafficClass = trafficClass;
		return this;
	}
	public int getFlowLabel() {
		return flowLabel;
	}
	public IPv6 setFlowLabel(int flowLabel) {
		this.flowLabel = flowLabel;
		return this;
	}
	public short getPayloadLength() {
		return payloadLength;
	}
	public IPv6 setPayloadLength(short payloadLength) {
		this.payloadLength = payloadLength;
		return this;
	}
	public IpProtocol getNextHeader() {
		return nextHeader;
	}
	public IPv6 setNextHeader(IpProtocol nextHeader) {
		this.nextHeader = nextHeader;
		return this;
	}
	public byte getHopLimit() {
		return hopLimit;
	}
	public IPv6 setHopLimit(byte hopLimit) {
		this.hopLimit = hopLimit;
		return this;
	}
	public IPv6Address getSourceAddress() {
		return sourceAddress;
	}
	public IPv6 setSourceAddress(IPv6Address sourceAddress) {
		this.sourceAddress = sourceAddress;
		return this;
	}
	public IPv6Address getDestinationAddress() {
		return destinationAddress;
	}
	public IPv6 setDestinationAddress(IPv6Address destinationAddress) {
		this.destinationAddress = destinationAddress;
		return this;
	}
	@Override
	public byte[] serialize() {
		byte[] payloadData = null;
		if (this.payload != null) {
			this.payload.setParent(this);
			payloadData = this.payload.serialize();
			if ((this.nextHeader == null || this.nextHeader.equals(IpProtocol.NONE))) {
				if (IPv6.nextHeaderClassMap.containsValue(this.payload.getClass())) {
					Set<Map.Entry<IpProtocol, Class<? extends IPacket>>> entries = IPv6.nextHeaderClassMap.entrySet();
					for (Map.Entry<IpProtocol, Class<? extends IPacket>> m : entries) {
						if (m.getValue().equals(this.payload.getClass())) {            		
							this.setNextHeader(m.getKey());
							log.warn("Setting previously unset IPv6 'next header' to {} as detected by payload {}", m.getKey(), this.getPayload().getClass().toString());
							break;
						}
					}
				} else if (this.payload instanceof Data) {
				} else {
					throw new IllegalArgumentException("IpProtocol is unset in IPv6 packet " + this.toString() + ". Unable to determine payload type to set for payload " + this.getPayload().getClass().toString());
				}
			}
		}
		this.payloadLength = (short) ((payloadData != null) ? payloadData.length : 0);
		byte[] data = new byte[HEADER_LENGTH + this.payloadLength];
		ByteBuffer bb = ByteBuffer.wrap(data);
		bb.put((byte) (((this.version & 0xF) << 4) |
				((this.trafficClass & 0xF0) >>> 4)));
		bb.put((byte) (((this.trafficClass & 0xF) << 4) |
				((this.flowLabel & 0xF0000) >>> 16)));
		bb.putShort((short) (this.flowLabel & 0xFFFF));
		bb.putShort(this.payloadLength);
		bb.put((byte) this.nextHeader.getIpProtocolNumber());
		bb.put(this.hopLimit);
		bb.put(this.sourceAddress.getBytes());
		bb.put(this.destinationAddress.getBytes());
		if (payloadData != null)
			bb.put(payloadData);
		return data;
	}
	@Override
	public String toString() {
		return "IPv6 [version=" + version + ", trafficClass=" + trafficClass
				+ ", flowLabel=" + flowLabel + ", payloadLength="
				+ payloadLength + ", nextHeader=" + nextHeader + ", hopLimit="
				+ hopLimit + ", sourceAddress=" + sourceAddress
				+ ", destinationAddress=" + destinationAddress + ", parent="
				+ parent + ", payload=" + payload + "]";
	}
	@Override
	public IPacket deserialize(byte[] data, int offset, int length)
			throws PacketParsingException {
		ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
		byte firstByte = bb.get();
		byte secondByte = bb.get();
		this.version = (byte) ((firstByte & 0xF0) >>> 4);
		if (this.version != 6) {
			throw new PacketParsingException(
					"Invalid version for IPv6 packet: " +
							this.version);
		}
		this.trafficClass = (byte) (((firstByte & 0xF) << 4) |
				((secondByte & 0xF0) >>> 4));
		this.flowLabel = ((secondByte & 0xF) << 16) |
				(bb.getShort() & 0xFFFF);
		this.payloadLength = bb.getShort();
		this.nextHeader = IpProtocol.of(bb.get());
		this.hopLimit = bb.get();
		byte[] sourceAddress = new byte[16];
		bb.get(sourceAddress, 0, 16);
		byte[] destinationAddress = new byte[16];
		bb.get(destinationAddress, 0, 16);
		this.sourceAddress = IPv6Address.of(sourceAddress);
		this.destinationAddress = IPv6Address.of(destinationAddress);
		IPacket payload;
		if (IPv6.nextHeaderClassMap.containsKey(this.nextHeader)) {
			Class<? extends IPacket> clazz = IPv6.nextHeaderClassMap.get(this.nextHeader);
			try {
				payload = clazz.newInstance();
			} catch (Exception e) {
				throw new RuntimeException("Error parsing payload for IPv6 packet", e);
			}
		} else {
			payload = new Data();
		}
		this.payload = payload.deserialize(data, bb.position(),
				Math.min(this.payloadLength, bb.limit() - bb.position()));
		this.payload.setParent(this);
		return this;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime
				+ ((destinationAddress == null) ? 0 : destinationAddress
						.hashCode());
				+ ((nextHeader == null) ? 0 : nextHeader.hashCode());
				+ ((sourceAddress == null) ? 0 : sourceAddress.hashCode());
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (!(obj instanceof IPv6))
			return false;
		IPv6 other = (IPv6) obj;
		if (destinationAddress == null) {
			if (other.destinationAddress != null)
				return false;
		} else if (!destinationAddress.equals(other.destinationAddress))
			return false;
		if (flowLabel != other.flowLabel)
			return false;
		if (hopLimit != other.hopLimit)
			return false;
		if (nextHeader == null) {
			if (other.nextHeader != null)
				return false;
		} else if (!nextHeader.equals(other.nextHeader))
			return false;
		if (payloadLength != other.payloadLength)
			return false;
		if (sourceAddress == null) {
			if (other.sourceAddress != null)
				return false;
		} else if (!sourceAddress.equals(other.sourceAddress))
			return false;
		if (trafficClass != other.trafficClass)
			return false;
		if (version != other.version)
			return false;
		return true;
	}
}
