package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.projectfloodlight.openflow.types.IPv6Address;
public class ICMPv6Option{
	protected byte code;
	protected byte length;
	protected byte prefixLength;
	protected boolean lBit;
	protected boolean aBit;
	protected int validLifetime;
	protected int preferredLifetime;
	protected IPv6Address prefixAddress;
	protected byte[] data;
	public ICMPv6Option(byte code,byte length){
		this.code = code;
		this.length = length;
	}
	public byte getCode() {
		return code;
	}
	public void setCode(byte code) {
		this.code = code;
	}
	public byte getLength() {
		return length;
	}
	public void setLength(byte length) {
		this.length = length;
	}
	public byte getPrefixLength() {
		return prefixLength;
	}
	public void setPrefixLength(byte prefixLength) {
		this.prefixLength = prefixLength;
	}
	public boolean islBit() {
		return lBit;
	}
	public void setlBit(boolean lBit) {
		this.lBit = lBit;
	}
	public boolean isaBit() {
		return aBit;
	}
	public void setaBit(boolean aBit) {
		this.aBit = aBit;
	}
	public int getValidLifetime() {
		return validLifetime;
	}
	public void setValidLifetime(int validLifetime) {
		this.validLifetime = validLifetime;
	}
	public int getPreferredLifetime() {
		return preferredLifetime;
	}
	public void setPreferredLifetime(int preferredLifetime) {
		this.preferredLifetime = preferredLifetime;
	}
	public IPv6Address getPrefixAddress() {
		return prefixAddress;
	}
	public void setPrefixAddress(IPv6Address prefixAddress) {
		this.prefixAddress = prefixAddress;
	}
	public byte[] getData() {
		return data;
	}
	public void setData(byte[] data){
		this.data=data;
	}
	public byte[] serilize(){
		return data;
	}
	public static ICMPv6Option getOption(byte[] data,int offset){
		ICMPv6Option option = new ICMPv6Option(data[offset],data[offset+1]);
		if(option.getCode()==3){
			ByteBuffer bb=ByteBuffer.wrap(data);
			bb.position(offset+2);
			option.setPrefixLength(bb.get());
			option.setlBit(((tmp>>7)&1) == 1);
			option.setaBit(((tmp>>6)&1) == 1);
			option.setValidLifetime(bb.getInt());
			option.setPreferredLifetime(bb.getInt());
			byte[] addressByte=new byte[16];
			for(int i=0;i<16;i++){
				addressByte[i]=bb.get();
			}
			option.setPrefixAddress(IPv6Address.of(addressByte));
		}
			option.data[i] = data[i + offset];
		}
		return option;
	}
	public static List<ICMPv6Option> getOptions(byte[] data,int offset){
		List<ICMPv6Option> options = new ArrayList<ICMPv6Option>();
		for(int n=offset;n<data.length;){
			ICMPv6Option option = getOption(data, n);
			options.add(option);
		}
		return options;
	}
	@Override
	public int hashCode() {
		int result=this.code;
		result+=this.prefixLength;
		result+=this.prefixAddress==null?0:this.prefixAddress.hashCode();
		result+=this.validLifetime;
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if(this == obj) return true;
		if(obj instanceof ICMPv6Option) {
			ICMPv6Option icmPv6Option=(ICMPv6Option)obj;
			if( icmPv6Option.getCode()==this.code&&icmPv6Option.getLength()==this.length) {
				if(icmPv6Option.getCode()==3) {
					return icmPv6Option.getPrefixAddress().equals(this.prefixAddress)
							&&icmPv6Option.getPrefixLength()==this.prefixLength
							&&icmPv6Option.getValidLifetime()==this.validLifetime;
				}else {
					return icmPv6Option.getData().equals(this.data);
				}
			}
		}
		return false;
	}
}
