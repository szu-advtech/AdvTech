package net.floodlightcontroller.dhcpserver;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
public class DHCPBinding {
	private MacAddress MAC = MacAddress.NONE;
	private IPv4Address IP = IPv4Address.NONE;
	private boolean LEASE_STATUS;
	private boolean PERMANENT_LEASE;
	private long LEASE_START_TIME_SECONDS;
	private long LEASE_DURATION_SECONDS;
	protected DHCPBinding(IPv4Address ip, MacAddress mac) {
		this.setMACAddress(mac);
		this.setIPv4Addresss(ip);
		this.setLeaseStatus(false);
	}
	public IPv4Address getIPv4Address() {
		return IP;
	}
	public MacAddress getMACAddress() {
		return MAC;
	}
	private void setIPv4Addresss(IPv4Address ip) {
		IP = ip; 
	}
	public void setMACAddress(MacAddress mac) {
		MAC = mac;
	}
	public boolean isActiveLease() {
		return LEASE_STATUS;
	}
	public void setStaticIPLease(boolean staticIP) {
		PERMANENT_LEASE = staticIP;
	}
	public boolean isStaticIPLease() {
		return PERMANENT_LEASE;
	}
	public void setLeaseStatus(boolean status) {
		LEASE_STATUS = status;
	}
	public boolean isLeaseExpired() {
		long currentTime = System.currentTimeMillis();
		if ((currentTime / 1000) >= (LEASE_START_TIME_SECONDS + LEASE_DURATION_SECONDS)) {
			return true;
		} else {
			return false;
		}
	}
	protected void setLeaseStartTimeSeconds() {
		LEASE_START_TIME_SECONDS = System.currentTimeMillis() / 1000;
	}
	protected void setLeaseDurationSeconds(long time) {
		LEASE_DURATION_SECONDS = time;
	}
	protected void clearLeaseTimes() {
		LEASE_START_TIME_SECONDS = 0;
		LEASE_DURATION_SECONDS = 0;
	}
	protected boolean cancelLease() {
		this.clearLeaseTimes();
		this.setLeaseStatus(false);
		return true;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime
				+ (int) (LEASE_DURATION_SECONDS ^ (LEASE_DURATION_SECONDS >>> 32));
		result = prime
				+ (int) (LEASE_START_TIME_SECONDS ^ (LEASE_START_TIME_SECONDS >>> 32));
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DHCPBinding other = (DHCPBinding) obj;
		if (IP == null) {
			if (other.IP != null)
				return false;
		} else if (!IP.equals(other.IP))
			return false;
		if (LEASE_DURATION_SECONDS != other.LEASE_DURATION_SECONDS)
			return false;
		if (LEASE_START_TIME_SECONDS != other.LEASE_START_TIME_SECONDS)
			return false;
		if (LEASE_STATUS != other.LEASE_STATUS)
			return false;
		if (MAC == null) {
			if (other.MAC != null)
				return false;
		} else if (!MAC.equals(other.MAC))
			return false;
		if (PERMANENT_LEASE != other.PERMANENT_LEASE)
			return false;
		return true;
	}
}