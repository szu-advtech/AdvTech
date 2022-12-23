package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.List;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
public class DHCPv6 extends BasePacket {
	protected DHCPv6MessageType msgType;
	protected int transactionId;
	protected List<DHCPv6Option> options;
	protected short duidType;
	protected short hardwareType;
	protected int time;
	protected MacAddress linkLayerAddress;
	protected int iaid;
	protected int t1;
	protected int t2;
	protected int validLifetime;
	protected int preferredLifetime;
	protected IPv6Address targetAddress;
	protected byte[] cache;
	@Override
	public byte[] serialize() {
		return cache;
	}
	@Override
	public IPacket deserialize(byte[] data, int offset, int length) throws PacketParsingException {
		cache = new byte[length];
		for(int i=0;i<length;i++){
			cache[i] = data[offset+i];
		}
		time = 0;
		validLifetime = 0;
		msgType = DHCPv6MessageType.getMsgType(data[offset]);
		transactionId = data[offset + 1];
		transactionId = ((transactionId<<8)&0xFF00) + data[offset + 2];
		transactionId = ((transactionId<<8)&0xFFFF00) + data[offset + 3];
		options = DHCPv6Option.getOptions(data, offset + 4);
		for(DHCPv6Option option:options){
			ByteBuffer bb = null;
			switch(option.getCode()){
			case DHCPv6Option.CLIENT_IDENTIFIER:
				bb = ByteBuffer.wrap(option.getData());
				duidType = bb.getShort();
				hardwareType = bb.getShort();
				time = bb.getInt();
				byte[] tmp = new byte[6];
				bb.get(tmp);
				linkLayerAddress = MacAddress.of(tmp);
				break;
			case DHCPv6Option.IDENTITY_ASSOCIATION:
				bb = ByteBuffer.wrap(option.getData());
				iaid = bb.getInt();
				t1 = bb.getInt();
				t2 = bb.getInt();
				if(option.length>12&&bb.getShort() == 5){
					bb.getShort();
					byte[] addr = new byte[16];
					bb.get(addr);
					targetAddress = IPv6Address.of(addr);
					preferredLifetime = bb.getInt();
					validLifetime = bb.getInt();
				}
				break;
			default:
				break;
			}
		}
		return this;
	}
	public int getTime() {
		return time;
	}
	public int getValidLifetime() {
		return validLifetime;
	}
	public int getIaid() {
		return iaid;
	}
	public void setIaid(int iaid) {
		this.iaid = iaid;
	}
	public int getT1() {
		return t1;
	}
	public void setT1(int t1) {
		this.t1 = t1;
	}
	public int getT2() {
		return t2;
	}
	public void setT2(int t2) {
		this.t2 = t2;
	}
	public IPv6Address getTargetAddress() {
		return targetAddress;
	}
	public short getDuidType() {
		return duidType;
	}
	public short getHardwareType() {
		return hardwareType;
	}
	public MacAddress getLinkLayerAddress() {
		return linkLayerAddress;
	}
	public DHCPv6MessageType getMessageType(){
		return msgType;
	}
	public int getTransactionId() {
		return transactionId;
	}
	public void setTransactionId(int transactionId) {
		this.transactionId = transactionId;
	}
}
