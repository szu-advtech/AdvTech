package net.floodlightcontroller.savi.binding;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPAddress;
import org.projectfloodlight.openflow.types.IPVersion;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import net.floodlightcontroller.devicemanager.SwitchPort;
public class BindingManager {
	protected Map<IPv4Address, Binding<IPv4Address>> ipv4Binding;
	protected Map<IPv6Address, Binding<IPv6Address>> ipv6Binding;
	protected Map<MacAddress, SwitchPort> hardwareBinding;
	protected Map<DatapathId, Set<IPv4Address>> ipv4SwitchBinding;
	protected Map<DatapathId, Set<IPv6Address>> ipv6SwitchBinding;
	protected Map<DatapathId, Set<MacAddress>> hardwareSwitchBinding;
	public static final byte SUCCESS = 0;
	public static final byte FAIL = 1;
	public static final byte NO_EXIST = 2;
	public BindingManager() {
		ipv4Binding = new ConcurrentHashMap<>();
		ipv6Binding = new ConcurrentHashMap<>();
		hardwareBinding = new ConcurrentHashMap<>();
		ipv4SwitchBinding = new ConcurrentHashMap<>();
		ipv6SwitchBinding = new ConcurrentHashMap<>();
		hardwareSwitchBinding = new ConcurrentHashMap<>();
	}
	public void addSwitch(DatapathId switchId) {
		if(!ipv4SwitchBinding.containsKey(switchId)){
			ipv4SwitchBinding.put(switchId, new HashSet<IPv4Address>());
		}
		if(!ipv6SwitchBinding.containsKey(switchId)){
			ipv6SwitchBinding.put(switchId, new HashSet<IPv6Address>());
		}
		if(!hardwareSwitchBinding.containsKey(switchId)){
			hardwareSwitchBinding.put(switchId, new HashSet<MacAddress>());
		}
	}
	@SuppressWarnings("unchecked")
	public void addBinding(Binding<?> binding) {
		IPAddress<?> address = binding.getAddress();
		SwitchPort switchPort = binding.getSwitchPort();
		DatapathId switchId = switchPort.getSwitchDPID();
		MacAddress macAddress = binding.getMacAddress();
		if(address.getIpVersion() == IPVersion.IPv4) {
			IPv4Address ipv4Address = (IPv4Address) address;
			ipv4Binding.put(ipv4Address, (Binding<IPv4Address>)binding);
			Set<IPv4Address> set = null;
			if(ipv4SwitchBinding.containsKey(switchId)){
				set = ipv4SwitchBinding.get(switchId);
			}
			else{
				set = new HashSet<>();
				ipv4SwitchBinding.put(switchId, set);
			}
			set.add(ipv4Address);
		}
		else {
			IPv6Address ipv6Address = (IPv6Address) address;
			ipv6Binding.put(ipv6Address, (Binding<IPv6Address>)binding);
			Set<IPv6Address> set = null;
			if(ipv6SwitchBinding.containsKey(switchId)){
				set = ipv6SwitchBinding.get(switchId);
			}
			else{
				set = new HashSet<>();
				ipv6SwitchBinding.put(switchId, set);
			}
			set.add(ipv6Address);
		}
		addBinding(macAddress, binding.getSwitchPort());
	}
	public void addBinding(MacAddress macAddress, SwitchPort switchPort) {
		if (!hardwareBinding.containsKey(macAddress)) {
			hardwareBinding.put(macAddress, switchPort);
			DatapathId dpid = switchPort.getSwitchDPID();
			Set<MacAddress> set = null;
			if(hardwareSwitchBinding.containsKey(dpid)){
				set = hardwareSwitchBinding.get(dpid);
			}
			else{
				set = new HashSet<>();
			}
			set.add(macAddress);
		}
	}
	public SwitchPort getSwitchPort(IPv4Address ipv4Address) {
		Binding<?> binding = ipv4Binding.get(ipv4Address);
		if (binding != null) {
			return binding.getSwitchPort();
		}
		return null;
	}
	public SwitchPort getSwitchPort(IPv6Address ipv6Address) {
		Binding<?> binding = ipv6Binding.get(ipv6Address);
		if(binding != null){
			return binding.getSwitchPort();
		}
		return null;
	}
	public boolean check(SwitchPort switchPort, IPAddress<?> address) {
		Binding<?> binding = null;
		if(address.getIpVersion() == IPVersion.IPv4) {
			binding = ipv4Binding.get(address);
		}
		else {
			binding = ipv6Binding.get(address);
		}
		if(binding == null) {
			return false;
		}
		if(binding.getSwitchPort().equals(switchPort)) {
			return true;
		}
		else {
			return false;
		}
	}
	public boolean check(SwitchPort switchPort, MacAddress macAddress, IPAddress<?> address) {
		Binding<?> binding = null;
		if(address.getIpVersion() == IPVersion.IPv4){
			binding = ipv4Binding.get(address);
			System.out.println("ipv4 binding");
		}
		else {
			binding = ipv6Binding.get(address);
		}
		if(binding == null) {
			System.out.println("binding == null");
			return false;
		}
		else {
			if (binding.getMacAddress().equals(macAddress) && binding.getSwitchPort().equals(switchPort)) {
				return true;
			} else {
				System.out.println(binding.getMacAddress().toString()+"   return false   "+binding.getSwitchPort().toString());
				return false;
			}
		}
	}
	public void delBinding(IPAddress<?> address) {
		Binding<?> binding = null;
		if(address.getIpVersion() == IPVersion.IPv4) {
			binding = ipv4Binding.remove(address);
		}
		else {
			binding =  ipv6Binding.remove(address);
		}
		if(binding!=null) {
			DatapathId switchId = binding.getSwitchPort().getSwitchDPID();
			if(address.getIpVersion() == IPVersion.IPv4){
				ipv4SwitchBinding.get(switchId).remove(address);
			}
			else {
				ipv6SwitchBinding.get(switchId).remove(address);
			}
		}
	}
	public MacAddress getMacAddress(IPAddress<?> address) {
		Binding<?> binding = null;
		if(address.getIpVersion() == IPVersion.IPv4){
			binding = ipv4Binding.get(address);
		}
		else {
			binding = ipv6Binding.get(address);
		}
		if(binding != null){
			return binding.getMacAddress();
		}
		return null;
	}
	public Set<IPv4Address> getIPv4SwitchBinding(DatapathId dpid){
		return ipv4SwitchBinding.get(dpid);
	}
	public Set<IPv6Address> getIPv6SwitchBinding(DatapathId dpid){
		return ipv6SwitchBinding.get(dpid);
	}
	public Set<MacAddress> getHardwareBinding(DatapathId dpid){
		return hardwareSwitchBinding.get(dpid);
	}
	public void removeSwitch(DatapathId dpid){
		Set<IPv4Address> ipv4Set = ipv4SwitchBinding.get(dpid);
		Set<IPv6Address> ipv6Set = ipv6SwitchBinding.get(dpid);
		if(ipv4Set!=null){
			for(IPv4Address ipv4Address:ipv4Set){
				ipv4Binding.remove(ipv4Address);
			}
			ipv4SwitchBinding.remove(dpid);
		}
		if(ipv6Set!=null){
			for(IPv6Address ipv6Address:ipv6Set){
				ipv6Binding.remove(ipv6Address);
			}
			ipv6SwitchBinding.remove(dpid);
		}
	}
	public List<Binding<?>> getBindings(){
		List<Binding<?>> bindingList = new ArrayList<>();
		bindingList.addAll(ipv4Binding.values());
		bindingList.addAll(ipv6Binding.values());
		return bindingList;
	}
}
