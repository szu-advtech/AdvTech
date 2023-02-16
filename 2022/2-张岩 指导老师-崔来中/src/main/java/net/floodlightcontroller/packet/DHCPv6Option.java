package net.floodlightcontroller.packet;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
public class DHCPv6Option {
	protected short code;
	protected short length;
	protected byte[] data;
	public static final short CLIENT_IDENTIFIER = 1;
	public static final short IDENTITY_ASSOCIATION = 3;
	public DHCPv6Option(short code,short length){
		this.code = code;
		this.length = length;
		this.data = new byte[length];
	}
	public short getCode() {
		return code;
	}
	public void setCode(short code) {
		this.code = code;
	}
	public void setData(byte[] data){
		this.data = data;
	}
	public  byte[] getData(){
		return data;
	}
	public short getLength() {
		return length;
	}
	public void setLength(byte length) {
		this.length = length;
	}
	public static DHCPv6Option getOption(byte[] data,int offset){
		short length = 0;
		short code = 0; 
		if(offset>= data.length){
			return null;
		}
		ByteBuffer bb = ByteBuffer.wrap(data,offset,4);
		code = bb.getShort();
		length = bb.getShort();
		if(code == 0xff00){
			return null;
		}
		DHCPv6Option option = new DHCPv6Option(code, length); 
		for(int i=0;i<length;i++){
			option.data[i] = data[offset+i+4];
		}
		return option;
	}
	public static List<DHCPv6Option> getOptions(byte[] data,int offset){
		List<DHCPv6Option> options = new ArrayList<DHCPv6Option>();
		DHCPv6Option option = getOption(data, offset);
		while(option!=null){
			options.add(option);
			offset += option.length + 4;
			option = getOption(data, offset);
		}
		return options;
	}
	public byte[] serilize(){
		byte[] data = new byte[4 + length];
		ByteBuffer bb = ByteBuffer.wrap(data);
		bb.putShort(code);
		bb.putShort(length);
		bb.put(this.data);
		return data;
	}
}
