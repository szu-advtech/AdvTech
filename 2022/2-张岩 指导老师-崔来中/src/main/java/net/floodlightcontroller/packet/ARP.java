package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import org.projectfloodlight.openflow.types.ArpOpcode;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
public class ARP extends BasePacket {
	public static short HW_TYPE_ETHERNET = 0x1;
	public static short PROTO_TYPE_IP = 0x800;
	public static ArpOpcode OP_REQUEST = ArpOpcode.REQUEST;
	public static ArpOpcode OP_REPLY = ArpOpcode.REPLY;
	public static ArpOpcode OP_RARP_REQUEST = ArpOpcode.REQUEST_REVERSE;
	public static ArpOpcode OP_RARP_REPLY = ArpOpcode.REPLY_REVERSE;
	protected short hardwareType;
	protected short protocolType;
	protected byte hardwareAddressLength;
	protected byte protocolAddressLength;
	protected ArpOpcode opCode;
	protected MacAddress senderHardwareAddress;
	protected IPv4Address senderProtocolAddress;
	protected MacAddress targetHardwareAddress;
	protected IPv4Address targetProtocolAddress;
	public short getHardwareType() {
		return hardwareType;
	}
	public ARP setHardwareType(short hardwareType) {
		this.hardwareType = hardwareType;
		return this;
	}
	public short getProtocolType() {
		return protocolType;
	}
	public ARP setProtocolType(short protocolType) {
		this.protocolType = protocolType;
		return this;
	}
	public byte getHardwareAddressLength() {
		return hardwareAddressLength;
	}
	public ARP setHardwareAddressLength(byte hardwareAddressLength) {
		this.hardwareAddressLength = hardwareAddressLength;
		return this;
	}
	public byte getProtocolAddressLength() {
		return protocolAddressLength;
	}
	public ARP setProtocolAddressLength(byte protocolAddressLength) {
		this.protocolAddressLength = protocolAddressLength;
		return this;
	}
	public ArpOpcode getOpCode() {
		return opCode;
	}
	public ARP setOpCode(ArpOpcode opCode) {
		this.opCode = opCode;
		return this;
	}
	public MacAddress getSenderHardwareAddress() {
		return senderHardwareAddress;
	}
	public ARP setSenderHardwareAddress(MacAddress senderHardwareAddress) {
		this.senderHardwareAddress = senderHardwareAddress;
		return this;
	}
	public IPv4Address getSenderProtocolAddress() {
		return senderProtocolAddress;
	}
	public ARP setSenderProtocolAddress(IPv4Address senderProtocolAddress) {
		this.senderProtocolAddress = senderProtocolAddress;
		return this;
	}
	public MacAddress getTargetHardwareAddress() {
		return targetHardwareAddress;
	}
	public ARP setTargetHardwareAddress(MacAddress targetHardwareAddress) {
		this.targetHardwareAddress = targetHardwareAddress;
		return this;
	}
	public IPv4Address getTargetProtocolAddress() {
		return targetProtocolAddress;
	}
	public boolean isGratuitous() {        
		assert(senderProtocolAddress.getLength() == targetProtocolAddress.getLength());
		return senderProtocolAddress.equals(targetProtocolAddress);
	}
	public ARP setTargetProtocolAddress(IPv4Address targetProtocolAddress) {
		this.targetProtocolAddress = targetProtocolAddress;
		return this;
	}
	@Override
	public byte[] serialize() {
		byte[] data = new byte[length];
		ByteBuffer bb = ByteBuffer.wrap(data);
		bb.putShort(this.hardwareType);
		bb.putShort(this.protocolType);
		bb.put(this.hardwareAddressLength);
		bb.put(this.protocolAddressLength);
		bb.putShort((short) this.opCode.getOpcode());
		bb.put(this.senderHardwareAddress.getBytes());
		bb.put(this.senderProtocolAddress.getBytes());
		bb.put(this.targetHardwareAddress.getBytes());
		bb.put(this.targetProtocolAddress.getBytes());
		return data;
	}
	@Override
	public IPacket deserialize(byte[] data, int offset, int length)
			throws PacketParsingException {
		ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
		this.hardwareType = bb.getShort();
		this.protocolType = bb.getShort();
		this.hardwareAddressLength = bb.get();
		this.protocolAddressLength = bb.get();
		if (this.hardwareAddressLength != 6) {
			String msg = "Incorrect ARP hardware address length: " +
					hardwareAddressLength;
			throw new PacketParsingException(msg);
		}
		if (this.protocolAddressLength != 4) {
			String msg = "Incorrect ARP protocol address length: " +
					protocolAddressLength;
			throw new PacketParsingException(msg);
		}
		this.opCode = ArpOpcode.of(bb.getShort());
		byte[] tmpMac = new byte[0xff & this.hardwareAddressLength];
		byte[] tmpIp = new byte[0xff & this.protocolAddressLength];
		bb.get(tmpMac, 0, this.hardwareAddressLength);
		this.senderHardwareAddress = MacAddress.of(tmpMac);  
		bb.get(tmpIp, 0, this.protocolAddressLength);
		this.senderProtocolAddress = IPv4Address.of(tmpIp);
		bb.get(tmpMac, 0, this.hardwareAddressLength);
		this.targetHardwareAddress = MacAddress.of(tmpMac);  
		bb.get(tmpIp, 0, this.protocolAddressLength);
		this.targetProtocolAddress = IPv4Address.of(tmpIp);
		return this;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime
				+ ((senderHardwareAddress == null) ? 0 : senderHardwareAddress
						.hashCode());
		result = prime
				+ ((senderProtocolAddress == null) ? 0 : senderProtocolAddress
						.hashCode());
		result = prime
				+ ((targetHardwareAddress == null) ? 0 : targetHardwareAddress
						.hashCode());
		result = prime
				+ ((targetProtocolAddress == null) ? 0 : targetProtocolAddress
						.hashCode());
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (getClass() != obj.getClass())
			return false;
		ARP other = (ARP) obj;
		if (hardwareAddressLength != other.hardwareAddressLength)
			return false;
		if (hardwareType != other.hardwareType)
			return false;
		if (opCode == null) {
			if (other.opCode != null)
				return false;
		} else if (!opCode.equals(other.opCode))
			return false;
		if (protocolAddressLength != other.protocolAddressLength)
			return false;
		if (protocolType != other.protocolType)
			return false;
		if (senderHardwareAddress == null) {
			if (other.senderHardwareAddress != null)
				return false;
		} else if (!senderHardwareAddress.equals(other.senderHardwareAddress))
			return false;
		if (senderProtocolAddress == null) {
			if (other.senderProtocolAddress != null)
				return false;
		} else if (!senderProtocolAddress.equals(other.senderProtocolAddress))
			return false;
		if (targetHardwareAddress == null) {
			if (other.targetHardwareAddress != null)
				return false;
		} else if (!targetHardwareAddress.equals(other.targetHardwareAddress))
			return false;
		if (targetProtocolAddress == null) {
			if (other.targetProtocolAddress != null)
				return false;
		} else if (!targetProtocolAddress.equals(other.targetProtocolAddress))
			return false;
		return true;
	}
	@Override
	public String toString() {
		return "ARP [hardwareType=" + hardwareType + ", protocolType="
				+ protocolType + ", hardwareAddressLength="
				+ hardwareAddressLength + ", protocolAddressLength="
				+ protocolAddressLength + ", opCode=" + opCode
				+ ", senderHardwareAddress=" + senderHardwareAddress
				+ ", senderProtocolAddress=" + senderProtocolAddress
				+ ", targetHardwareAddress=" + targetHardwareAddress
				+ ", targetProtocolAddress=" + targetProtocolAddress + "]";
	}
}
