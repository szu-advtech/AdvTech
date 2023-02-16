package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.sdnplatform.sync.thrift.EchoRequestMessage;
public class ICMPv6 extends BasePacket {
	protected byte icmpv6Type;
	protected byte icmpv6Code;
	protected short checksum;
	protected List<ICMPv6Option> options = null;
	protected boolean routerFlag;
	protected boolean solicitedFlag;
	protected boolean overrideFlag;
	protected IPv6Address targetAddress;
	protected byte curHopLimit;
	protected boolean managedAddressConfiguration;
	protected boolean otherConfiguration;
	protected short routerLifeTime;
	protected int reachableTime;
	protected int retransTime;
	protected byte[] echoRequestData;
	protected byte[] cache;
	protected static final Map<Byte, Integer> paddingMap;
	private static final byte ROUTER_FLAG_MASK   = (byte)0x80;
	private static final byte SOLIITED_FLAG_MASK = (byte)0x40;
	private static final byte OVERRIDE_FLAG_MASK = (byte)0x20;
	public static final byte DESTINATION_UNREACHABLE = 1;
	public static final byte PACKET_TOO_BIG = 2;
	public static final byte TIME_EXCEEDED = 3;
	public static final byte PARAMETER_PROBLEM = 4;
	public static final byte ECHO_REQUEST = (byte)128;
	public static final byte ECHO_REPLY = (byte)129;
	public static final byte ROUTER_SOLICITATION = (byte)133;
	public static final byte ROUTER_ADVERTSEMENT = (byte)134;
	public static final byte NEIGHBOR_SOLICITATION = (byte)135;
	public static final byte NEIGHBOR_ADVERTISEMENT = (byte)136;
	static{
		paddingMap = new HashMap<Byte,Integer>();
		paddingMap.put(DESTINATION_UNREACHABLE, 0x4);
		paddingMap.put(PACKET_TOO_BIG, 0x4);
		paddingMap.put(TIME_EXCEEDED, 0x4);
		paddingMap.put(PARAMETER_PROBLEM, 0x4);
		paddingMap.put(ECHO_REQUEST, 0x4);
		paddingMap.put(ECHO_REPLY, 0x4);
		paddingMap.put(ROUTER_ADVERTSEMENT, 0x4);
		paddingMap.put(ROUTER_SOLICITATION, 0x4);
		paddingMap.put(NEIGHBOR_ADVERTISEMENT, 0x4);
		paddingMap.put(NEIGHBOR_SOLICITATION, 0x4);
	}
	public IPv6Address getTargetAddress() {
		return targetAddress;
	}
	public void setTargetAddress(IPv6Address targetAddress) {
		this.targetAddress = targetAddress;
	}
	public byte getICMPv6Type() {
		return icmpv6Type;
	}
	public void setICMPv6Type(byte icmpv6Type) {
		this.icmpv6Type = icmpv6Type;
	}
	public byte getICMPv6Code() {
		return icmpv6Code;
	}
	public void setICMPv6Code(byte icmpv6Code) {
		this.icmpv6Code = icmpv6Code;
	}
	public short getChecksum() {
		return checksum;
	}
	public void setChecksum(short checksum) {
		this.checksum = checksum;
	}
	public boolean isRouterFlag() {
		return routerFlag;
	}
	public void setRouterFlag(boolean routerFlag) {
		this.routerFlag = routerFlag;
	}
	public boolean isSolicitedFlag() {
		return solicitedFlag;
	}
	public void setSolicitedFlag(boolean solicitedFlag) {
		this.solicitedFlag = solicitedFlag;
	}
	public boolean isOverrideFlag() {
		return overrideFlag;
	}
	public void setOverrideFlag(boolean overrideFlag) {
		this.overrideFlag = overrideFlag;
	}
	public boolean isManagedAddressConfiguration() {
		return managedAddressConfiguration;
	}
	public void setManagedAddressConfiguration(boolean managedAddressConfiguration) {
		this.managedAddressConfiguration = managedAddressConfiguration;
	}
	public boolean isOtherConfiguration() {
		return otherConfiguration;
	}
	public void setOtherConfiguration(boolean otherConfiguration) {
		this.otherConfiguration = otherConfiguration;
	}
	public List<ICMPv6Option> getOptions() {
		return options;
	}
	public void setOptions(List<ICMPv6Option> options){ this.options = options; }
	@Override
	public int hashCode() {
        final int prime = 5807;
        int result = super.hashCode();
        return result;
	}
	@Override
	public byte[] serialize() {
		int len=0;
		if (this.options != null) {
			for(ICMPv6Option icmPv6Option : this.options){
			}
		}
		if(cache!=null)
			return cache;
		byte[] data=new byte[24+len];
		ByteBuffer bb=ByteBuffer.wrap(data);
		bb.put(this.icmpv6Type);
		bb.put(this.icmpv6Code);
		bb.putShort(this.checksum);
		if (this.icmpv6Type == ICMPv6.NEIGHBOR_SOLICITATION) {
			bb.putInt(0);
			bb.put(this.targetAddress.getBytes());
			if (this.options != null) {
				for(ICMPv6Option icmPv6Option : options){
					bb.put(icmPv6Option.serilize());
				}
			}
		}else if(this.icmpv6Type == ICMPv6.NEIGHBOR_ADVERTISEMENT){
			bb.putInt(3<<30);
			bb.put(targetAddress.getBytes());
		}else {
			bb.putShort((short)0x9223);
			bb.putShort((short) 1);
		}
		return data;
	}
	@Override
	public IPacket deserialize(byte[] data, int offset, int length) throws PacketParsingException {
		cache = new byte[length];	
		for(int i=0;i<length;i++){
			cache[i] = data[offset+i];
		}
		ByteBuffer bb = ByteBuffer.wrap(data, offset, length);
		this.icmpv6Type = bb.get();
		this.icmpv6Code = bb.get();
		this.checksum = bb.getShort();
		if(icmpv6Type!=NEIGHBOR_SOLICITATION&&icmpv6Type!=NEIGHBOR_ADVERTISEMENT
				&&icmpv6Type!=ROUTER_ADVERTSEMENT) {
			return this;
		}
		if(icmpv6Type == NEIGHBOR_ADVERTISEMENT){
			byte tmp = bb.get();
			this.routerFlag = ((tmp&ROUTER_FLAG_MASK)==ROUTER_FLAG_MASK);
			this.solicitedFlag = ((tmp&SOLIITED_FLAG_MASK)==SOLIITED_FLAG_MASK);
			this.overrideFlag = ((tmp&OVERRIDE_FLAG_MASK)==OVERRIDE_FLAG_MASK);
			bb.get();
			bb.getShort();
		}
		else if(icmpv6Type == NEIGHBOR_SOLICITATION){
			bb.getInt();
		}else if(icmpv6Type == ROUTER_ADVERTSEMENT){
			this.curHopLimit = bb.get();
			byte tmp=bb.get();
			this.managedAddressConfiguration = ((tmp>>7)&1) == 1;
			this.otherConfiguration = ((tmp>>6)&1) == 1;
			this.routerLifeTime=bb.getShort();
			this.reachableTime=bb.getInt();
			this.retransTime=bb.getInt();
		}
		if(icmpv6Type == NEIGHBOR_SOLICITATION||icmpv6Type == NEIGHBOR_ADVERTISEMENT){
			options = new ArrayList<>();
			byte[] tmp  = new byte[16];
			bb.get(tmp, 0, 16);
			targetAddress = IPv6Address.of(tmp);
		}
		this.options = ICMPv6Option.getOptions(data, bb.position());
		return this;
	}
}
