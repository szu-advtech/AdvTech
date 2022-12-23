package net.floodlightcontroller.savi.binding;
public enum BindingStatus{
	NO_ENTRY_EXIST(0,"NO_ENTRY_EXIST"),
	INIT(1,"INIT"),
	SELECTING(2,"SELECTING"),
	REQUESTING(3,"REQUESTING"),
	PRE_BOUND(4,"PRE_BOUND"),
	BOUND(5,"BOUND"),
	RENEWING(6,"RENEWING"),
	REBINDING(7,"REBINDING"),
	DETECTING(8,"DETECTING"),
	CONFIRMING(9,"CONFORMING");
	private int value;
	private String name;
	private BindingStatus(int value, String name){
		this.value = value; 
		this.name = name;
	}
	public int getValue(){
		return value;
	}
	@Override
	public String toString(){
		return name;
	}
	public static BindingStatus getStatus(int status){
		switch(status){
		case 1:
			return INIT;
		case 2:
			return SELECTING;
		case 3:
			return REQUESTING;
		case 4:
			return PRE_BOUND;
		case 5:
			return BOUND;
		case 6:
			return RENEWING;
		case 7:
			return REBINDING;
		default:
			return null;
		}
	}
}