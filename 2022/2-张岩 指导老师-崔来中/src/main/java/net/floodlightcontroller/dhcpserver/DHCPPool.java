package net.floodlightcontroller.dhcpserver;
import java.util.ArrayList;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.slf4j.Logger;
import net.floodlightcontroller.dhcpserver.DHCPBinding;
public class DHCPPool {
	protected Logger log;
	private volatile static ArrayList<DHCPBinding> DHCP_POOL = new ArrayList<DHCPBinding>();
	private volatile int POOL_SIZE;
	private volatile int POOL_AVAILABILITY;
	private volatile boolean POOL_FULL;
	private volatile IPv4Address STARTING_ADDRESS;
	private final MacAddress UNASSIGNED_MAC = MacAddress.NONE;
	public DHCPPool(IPv4Address startingIPv4Address, int size, Logger log) {
		this.log = log;
		int IPv4AsInt = startingIPv4Address.getInt();
		this.setPoolSize(size);
		this.setPoolAvailability(size);
		STARTING_ADDRESS = startingIPv4Address;
		for (int i = 0; i < size; i++) { 
			DHCP_POOL.add(new DHCPBinding(IPv4Address.of(IPv4AsInt + i), UNASSIGNED_MAC));
		}
	}
	private void setPoolFull(boolean full) {
		POOL_FULL = full;
	}
	private boolean isPoolFull() {
		return POOL_FULL;
	}
	private void setPoolSize(int size) {
		POOL_SIZE = size;
	}
	private int getPoolSize() {
		return POOL_SIZE;
	}
	private int getPoolAvailability() {
		return POOL_AVAILABILITY;
	}
	private void setPoolAvailability(int size) {
		POOL_AVAILABILITY = size;
	}
	public DHCPBinding getDHCPbindingFromIPv4(IPv4Address ip) {
		if (ip == null) return null;
		for (DHCPBinding binding : DHCP_POOL) {
			if (binding.getIPv4Address().equals(ip)) {
				return binding;
			}
		}
		return null;
	}
	public DHCPBinding getDHCPbindingFromMAC(MacAddress mac) {
		if (mac == null) return null;
		for (DHCPBinding binding : DHCP_POOL) {
			if (binding.getMACAddress().equals(mac)) {
				return binding;
			}
		}
		return null;
	}
	public boolean isIPv4Leased(IPv4Address ip) {
		DHCPBinding binding = this.getDHCPbindingFromIPv4(ip);
		if (binding != null) return binding.isActiveLease();
		else return false;
	}
	public void setDHCPbinding(DHCPBinding binding, MacAddress mac, int time) {
		int index = DHCP_POOL.indexOf(binding);
		binding.setMACAddress(mac);
		binding.setLeaseStatus(true);
		this.setPoolAvailability(this.getPoolAvailability() - 1);
		DHCP_POOL.set(index, binding);
		if (this.getPoolAvailability() == 0) setPoolFull(true);
		binding.setLeaseStartTimeSeconds();
		binding.setLeaseDurationSeconds(time);
	}
	public void removeIPv4FromDHCPPool(IPv4Address ip) {
		if (ip == null || getDHCPbindingFromIPv4(ip) == null) return;
		if (ip.equals(STARTING_ADDRESS)) {
			DHCPBinding lowest = null;
			for (DHCPBinding binding : DHCP_POOL) {
				if (lowest == null) {
					lowest = binding;
				} else if (binding.getIPv4Address().getInt() < lowest.getIPv4Address().getInt()
						&& !binding.getIPv4Address().equals(ip))
				{
					lowest = binding;
				}
			}
			STARTING_ADDRESS = lowest.getIPv4Address();
		}
		DHCP_POOL.remove(this.getDHCPbindingFromIPv4(ip));
		this.setPoolSize(this.getPoolSize() - 1);
		this.setPoolAvailability(this.getPoolAvailability() - 1);
		if (this.getPoolAvailability() == 0) this.setPoolFull(true);
	}
	public DHCPBinding addIPv4ToDHCPPool(IPv4Address ip) {
		DHCPBinding binding = null;
		if (this.getDHCPbindingFromIPv4(ip) == null) {
			if (ip.getInt() < STARTING_ADDRESS.getInt()) {
				STARTING_ADDRESS = ip;
			}
			binding = new DHCPBinding(ip, null);
			DHCP_POOL.add(binding);
			this.setPoolSize(this.getPoolSize() + 1);
			this.setPoolFull(false);
		}
		return binding;
	}
	public boolean hasAvailableAddresses() {
		if (isPoolFull() || getPoolAvailability() == 0) return false;
		else return true;
	}
	public DHCPBinding getAnyAvailableLease(MacAddress mac) {
		if (isPoolFull()) return null;
		DHCPBinding usedBinding = null;
		usedBinding = this.getDHCPbindingFromMAC(mac);
		if (usedBinding != null) return usedBinding;
		for (DHCPBinding binding : DHCP_POOL) {
			if (!binding.isActiveLease() 
					&& binding.getMACAddress().equals(UNASSIGNED_MAC))
			{
				return binding;
			} else if (!binding.isActiveLease() && usedBinding == null && !binding.isStaticIPLease()) {
				usedBinding = binding;
			}
		}
		return usedBinding;
	}
	public DHCPBinding getSpecificAvailableLease(IPv4Address ip, MacAddress mac) {
		if (ip == null || mac == null || isPoolFull()) return null;
		DHCPBinding binding = this.getDHCPbindingFromIPv4(ip);
		DHCPBinding binding2 = this.getDHCPbindingFromMAC(mac);
		if (binding2 != null && !binding2.isActiveLease() && binding2.isStaticIPLease() && binding != binding2) {
			if (log != null) log.info("Fixed DHCP entry for MAC trumps requested IP. Returning binding for MAC");
			return binding2;
		} else if (binding != null && !binding.isActiveLease() && binding.isStaticIPLease() && mac.equals(binding.getMACAddress())) {
			if (log != null) log.info("Found matching fixed DHCP entry for IP with MAC. Returning binding for IP with MAC");
			return binding;
		} else if (binding != null && !binding.isActiveLease() && !binding.isStaticIPLease()) {
			if (log != null) log.info("No fixed DHCP entry for IP or MAC found. Returning dynamic binding for IP.");
			return binding;
		} else {
			if (log != null) log.debug("Invalid IP address request or IP is actively leased...check for any available lease to resolve");
			return null;
		}
	}
	public boolean renewLease(IPv4Address ip, int time) {
		DHCPBinding binding = this.getDHCPbindingFromIPv4(ip);
		if (binding != null) {
			binding.setLeaseStartTimeSeconds();
			binding.setLeaseDurationSeconds(time);
			binding.setLeaseStatus(true);
			return true;
		}
		return false;
	}
	public boolean cancelLeaseOfIPv4(IPv4Address ip) {
		DHCPBinding binding = this.getDHCPbindingFromIPv4(ip);
		if (binding != null) {
			binding.clearLeaseTimes();
			binding.setLeaseStatus(false);
			this.setPoolAvailability(this.getPoolAvailability() + 1);
			this.setPoolFull(false);
			return true;
		}
		return false;
	}
	public boolean cancelLeaseOfMAC(MacAddress mac) {
		DHCPBinding binding = getDHCPbindingFromMAC(mac);
		if (binding != null) {
			binding.clearLeaseTimes();
			binding.setLeaseStatus(false);
			this.setPoolAvailability(this.getPoolAvailability() + 1);
			this.setPoolFull(false);
			return true;
		}
		return false;
	}
	public ArrayList<DHCPBinding> cleanExpiredLeases() {
		ArrayList<DHCPBinding> newAvailableLeases = new ArrayList<DHCPBinding>();
		for (DHCPBinding binding : DHCP_POOL) {
			if (binding.isLeaseExpired() && binding.isActiveLease()) {
				this.cancelLeaseOfIPv4(binding.getIPv4Address());
				this.setPoolAvailability(this.getPoolAvailability() + 1);
				this.setPoolFull(false);
				newAvailableLeases.add(binding);
			}
		}
		return newAvailableLeases;
	}
	public boolean configureFixedIPLease(IPv4Address ip, MacAddress mac) {
		DHCPBinding binding = this.getDHCPbindingFromIPv4(ip);
		if (binding != null) {
			binding.setMACAddress(mac);
			binding.setStaticIPLease(true);
			binding.setLeaseStatus(false);
			return true;
		} else {
			return false;
		}
	}
}