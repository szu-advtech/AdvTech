package net.floodlightcontroller.packet;
public enum DHCPv6MessageType {
	SOLICIT 			(1),
	ADVERTISE			(2),
	REQUEST 			(3),
	CONFIRM				(4),
	RENEW				(5),
	REBIND				(6),
	REPLY				(7),
	RELEASE				(8),
	DECLINE				(9),	
	RECONFIGURE			(10),
	INFORMATION_REQUEST	(11),
	RELAY_FORW 			(12),
	RELAY_REPL			(13);
	protected int value;
	private DHCPv6MessageType(int value) {
		this.value = value;
	}
	public String toString(){
		switch(this.value){
		case 1:
			return "SOLICIT";
		case 2:
			return "ADVERTISE";
		case 3:
			return "REQUEST";
		case 4:
			return "CONFIRM";
		case 5:
			return "RENEW";
		case 6:
			return "REBIND";
		case 7:
			return "REPLY";
		case 8:
			return "RELEASE";
		case 9:
			return "DECLINE";
		case 10:
			return "RECONFIGURE";
		case 11:
			return "INFORMATION_REQUEST";
		case 12:
			return "RELAY_FORW";
		case 13:
			return "RELAY_REPL";
		}
		return null;
	}
	public static DHCPv6MessageType getMsgType(int value){
		switch(value){
		case 1:
			return SOLICIT;
		case 2:
			return ADVERTISE;
		case 3:
			return REQUEST;
		case 4:
			return CONFIRM;
		case 5:
			return RENEW;
		case 6:
			return REBIND;
		case 7:
			return REPLY;
		case 8:
			return RELEASE;
		case 9:
			return DECLINE;
		case 10:
			return RECONFIGURE;
		case 11:
			return INFORMATION_REQUEST;
		case 12:
			return RELAY_FORW;
		case 13:
			return RELAY_REPL;
		}
		return null;
	}
	public byte getValue(){
		return (byte)value;
	}
}
