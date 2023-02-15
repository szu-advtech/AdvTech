package net.floodlightcontroller.savi.binding;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPAddress;
import org.projectfloodlight.openflow.types.MacAddress;
import net.floodlightcontroller.devicemanager.SwitchPort;
public class BindingPool<T extends IPAddress<?>>{
	protected Map<T, Binding<T>> bindingTable;
	protected Map<MacAddress, SwitchPort> hardwareBindingTable;
	public BindingPool() {
		bindingTable = new ConcurrentHashMap<>();
		hardwareBindingTable = new ConcurrentHashMap<>();
	}
	public boolean isBindingLeaseExpired(T address){
		Binding<T> binding = bindingTable.get(address);
		if(binding == null){
			return false;
		}
		else{
			return binding.isLeaseExpired();
		}
	}
	public void renewBiningLease(T address, int leaseTime){
		Binding<T> binding = bindingTable.get(address);
		if(binding != null){
			binding.setBindingTime();
			binding.setLeaseTime(leaseTime);
		}
	}
	public void addHardwareBinding(MacAddress macAddress,SwitchPort switchPort){
		synchronized(hardwareBindingTable){
			hardwareBindingTable.put(macAddress, switchPort);
		}
	}
	public void delHardwareBinding(MacAddress macAddress){
		hardwareBindingTable.remove(macAddress);
	}
	public boolean isContain(MacAddress macAddress){
		return hardwareBindingTable.containsKey(macAddress);
	}
	public boolean isContain(T address){
		return bindingTable.containsKey(address);
	}
	public SwitchPort getSwitchPort(MacAddress macAddress){
		return hardwareBindingTable.get(macAddress);
	}
	public Binding<T> getBinding(T address){
		return bindingTable.get(address);
	}
	public void addBinding(T address, Binding<T> binding){
		synchronized(bindingTable){
			bindingTable.put(address, binding);
		}
	}
	public void delBinding(T address){
		synchronized(bindingTable){
			bindingTable.remove(address);
		}
	}
	public void delSwitch(DatapathId switchId){
		synchronized(hardwareBindingTable){
			for(MacAddress macAddress:hardwareBindingTable.keySet()){
				SwitchPort switchPort = hardwareBindingTable.get(macAddress);
				if(switchId.equals(switchPort.getSwitchDPID())){
					hardwareBindingTable.remove(macAddress);
				}
			}
		}
		synchronized(bindingTable){
			for(T key:bindingTable.keySet()){
				SwitchPort switchPort = bindingTable.get(key).getSwitchPort();
				if(switchId.equals(switchPort.getSwitchDPID())){
					bindingTable.remove(key);
				}
			}
		}
	}
	public boolean check(MacAddress macAddress, SwitchPort switchPort){
		if(hardwareBindingTable.containsKey(macAddress)){
			SwitchPort tmp = hardwareBindingTable.get(macAddress);
			if(tmp.equals(switchPort)){
				return true;
			}
		}
		return false;
	}
	public boolean check(T address, MacAddress macAddress){
		if(bindingTable.containsKey(address)){
			Binding<T> tmp = bindingTable.get(address);
			if(macAddress.equals(tmp.getMacAddress())){
				return true;
			}
		}
		return false;
	}
	public boolean check(T address, MacAddress macAddress, SwitchPort switchPort){
		return check(macAddress, switchPort)&&check(address, macAddress);
	}
	public Collection<Binding<T>> getAllBindings(){
		return bindingTable.values();
	}
}
