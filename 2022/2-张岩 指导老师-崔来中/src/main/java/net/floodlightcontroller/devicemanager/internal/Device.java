package net.floodlightcontroller.devicemanager.internal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import net.floodlightcontroller.devicemanager.web.DeviceSerializer;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.devicemanager.SwitchPort.ErrorStatus;
import net.floodlightcontroller.topology.ITopologyService;
@JsonSerialize(using = DeviceSerializer.class)
public class Device implements IDevice {
	protected static Logger log = LoggerFactory.getLogger(Device.class);
	private final Long deviceKey;
	protected final DeviceManagerImpl deviceManager;
	protected final Entity[] entities;
	private final IEntityClass entityClass;
	protected final String macAddressString;
	protected final VlanVid[] vlanIds;
	protected volatile String dhcpClientName;
	protected volatile List<AttachmentPoint> oldAPs;
	protected volatile List<AttachmentPoint> attachmentPoints;
	public Device(DeviceManagerImpl deviceManager, Long deviceKey,
			Entity entity, IEntityClass entityClass) {
		this.deviceManager = deviceManager;
		this.deviceKey = deviceKey;
		this.entities = new Entity[] { entity };
		this.macAddressString = entity.getMacAddress().toString();
		this.entityClass = entityClass;
		Arrays.sort(this.entities);
		this.dhcpClientName = null;
		this.oldAPs = null;
		this.attachmentPoints = null;
		if (!entity.getSwitchDPID().equals(DatapathId.NONE)
				&& !entity.getSwitchPort().equals(OFPort.ZERO)) {
			DatapathId sw = entity.getSwitchDPID();
			OFPort port = entity.getSwitchPort();
			if (deviceManager.isValidAttachmentPoint(sw, port)) {
				AttachmentPoint ap;
				ap = new AttachmentPoint(sw, port,
						entity.getLastSeenTimestamp());
				this.attachmentPoints = new ArrayList<AttachmentPoint>();
				this.attachmentPoints.add(ap);
			}
		}
		vlanIds = computeVlandIds();
	}
	public Device(DeviceManagerImpl deviceManager, Long deviceKey,
			String dhcpClientName, Collection<AttachmentPoint> oldAPs,
			Collection<AttachmentPoint> attachmentPoints,
			Collection<Entity> entities, IEntityClass entityClass) {
		this.deviceManager = deviceManager;
		this.deviceKey = deviceKey;
		this.dhcpClientName = dhcpClientName;
		this.entities = entities.toArray(new Entity[entities.size()]);
		this.oldAPs = null;
		this.attachmentPoints = null;
		if (oldAPs != null) {
			this.oldAPs = new ArrayList<AttachmentPoint>(oldAPs);
		}
		if (attachmentPoints != null) {
			this.attachmentPoints = new ArrayList<AttachmentPoint>(
					attachmentPoints);
		}
		this.macAddressString = this.entities[0].getMacAddress().toString();
		this.entityClass = entityClass;
		Arrays.sort(this.entities);
		vlanIds = computeVlandIds();
	}
	public Device(Device device, Entity newEntity, int insertionpoint) {
		this.deviceManager = device.deviceManager;
		this.deviceKey = device.deviceKey;
		this.dhcpClientName = device.dhcpClientName;
		if (insertionpoint < 0) {
			insertionpoint = -(Arrays.binarySearch(device.entities, newEntity) + 1);
		}
		if (insertionpoint < 0) {
			log.warn("Performing a replacement upon new entity add in Device. Should the entity have been removed first?");
			this.entities = new Entity[device.entities.length];
			this.entities[replacementpoint] = newEntity;
			System.arraycopy(device.entities, replacementpoint + 1, this.entities, replacementpoint + 1, device.entities.length - (replacementpoint + 1));
		} else {
			this.entities = new Entity[device.entities.length + 1];
			if (insertionpoint > 0) {
				System.arraycopy(device.entities, 0, this.entities, 0,
						insertionpoint);
			}
			if (insertionpoint < device.entities.length) {
				System.arraycopy(device.entities, insertionpoint, this.entities,
						insertionpoint + 1, device.entities.length - insertionpoint);
			}
			this.entities[insertionpoint] = newEntity;
		}
		this.oldAPs = null;
		if (device.oldAPs != null) {
			this.oldAPs = new ArrayList<AttachmentPoint>(device.oldAPs);
		}
		this.attachmentPoints = null;
		if (device.attachmentPoints != null) {
			this.attachmentPoints = new ArrayList<AttachmentPoint>(
					device.attachmentPoints);
		}
		this.macAddressString = this.entities[0].getMacAddress().toString();
		this.entityClass = device.entityClass;
		vlanIds = computeVlandIds();
	}
	private VlanVid[] computeVlandIds() {
		if (entities.length == 1) {
			return new VlanVid[] { entities[0].getVlan() };
		}
		TreeSet<VlanVid> vals = new TreeSet<VlanVid>();
		for (Entity e : entities) {
			vals.add(e.getVlan());
		}
		return vals.toArray(new VlanVid[vals.size()]);
	}
	private Map<DatapathId, AttachmentPoint> getAPMap(
			List<AttachmentPoint> apList) {
		if (apList == null)
			return null;
		List<AttachmentPoint> oldAP = new ArrayList<AttachmentPoint>();
		if (apList != null) {
			oldAP.addAll(apList);
		}
		List<AttachmentPoint> tempAP = new ArrayList<AttachmentPoint>();
		for (AttachmentPoint ap : oldAP) {
			if (deviceManager.isValidAttachmentPoint(ap.getSw(), ap.getPort())) {
				tempAP.add(ap);
			}
		}
		oldAP = tempAP;
		Collections.sort(oldAP, deviceManager.apComparator);
		Map<DatapathId, AttachmentPoint> apMap = new HashMap<DatapathId, AttachmentPoint>();
		for (int i = 0; i < oldAP.size(); ++i) {
			AttachmentPoint ap = oldAP.get(i);
			if (!deviceManager.isValidAttachmentPoint(ap.getSw(), ap.getPort()))
				continue;
			DatapathId id = deviceManager.topology.getOpenflowDomainId(ap.getSw());
			apMap.put(id, ap);
		}
		if (apMap.isEmpty())
			return null;
		return apMap;
	}
	private boolean removeExpiredAttachmentPoints(List<AttachmentPoint> apList) {
		List<AttachmentPoint> expiredAPs = new ArrayList<AttachmentPoint>();
		if (apList == null)
			return false;
		for (AttachmentPoint ap : apList) {
			if (ap.getLastSeen().getTime()
					+ AttachmentPoint.INACTIVITY_INTERVAL < System
						.currentTimeMillis()) {
				expiredAPs.add(ap);
			}
		}
		if (expiredAPs.size() > 0) {
			apList.removeAll(expiredAPs);
			return true;
		} else
			return false;
	}
	List<AttachmentPoint> getDuplicateAttachmentPoints(
			List<AttachmentPoint> oldAPList,
			Map<DatapathId, AttachmentPoint> apMap) {
		ITopologyService topology = deviceManager.topology;
		List<AttachmentPoint> dupAPs = new ArrayList<AttachmentPoint>();
		long timeThreshold = System.currentTimeMillis() - AttachmentPoint.INACTIVITY_INTERVAL;
		if (oldAPList == null || apMap == null) {
			return dupAPs;
		}
		Set<DatapathId> visitedIslands = new HashSet<DatapathId>();
		for (AttachmentPoint ap : oldAPList) {
			DatapathId id = topology.getOpenflowDomainId(ap.getSw());
			AttachmentPoint trueAP = apMap.get(id);
			if (trueAP == null) {
				continue;
			}
			boolean c = (topology.isConsistent(trueAP.getSw(),
					trueAP.getPort(), ap.getSw(), ap.getPort()));
			boolean active = trueAP.getActiveSince().after(ap.getActiveSince())
					&& ap.getLastSeen().after(trueAP.getLastSeen());
			boolean last = ap.getLastSeen().getTime() > timeThreshold;
			if (!c && active && last) {
				visitedIslands.add(id);
			}
		}
		for (AttachmentPoint ap : oldAPList) {				
			DatapathId id = topology.getOpenflowDomainId(ap.getSw());
			if (visitedIslands.contains(id)) {
				if (ap.getLastSeen().getTime() > timeThreshold) {
					dupAPs.add(ap);
				}
			}
		}
		return dupAPs;
	}
	protected boolean updateAttachmentPoint() {
		boolean moved = false;
		this.oldAPs = attachmentPoints;
		if (attachmentPoints == null || attachmentPoints.isEmpty()) {
			return false;
		}
		List<AttachmentPoint> apList = new ArrayList<AttachmentPoint>();
		if (attachmentPoints != null) {
			apList.addAll(attachmentPoints);
		}
		Map<DatapathId, AttachmentPoint> newMap = getAPMap(apList);
		if (newMap == null || newMap.size() != apList.size()) {
			moved = true;
		}
		if (moved) {
			log.info("updateAttachmentPoint: ap {}  newmap {} ",
					attachmentPoints, newMap);
			List<AttachmentPoint> newAPList = new ArrayList<AttachmentPoint>();
			if (newMap != null) {
				newAPList.addAll(newMap.values());
			}
			this.attachmentPoints = newAPList;
		}
		return moved;
	}
	protected boolean updateAttachmentPoint(DatapathId sw, OFPort port,
			Date lastSeen) {
		ITopologyService topology = deviceManager.topology;
		List<AttachmentPoint> oldAPList;
		List<AttachmentPoint> apList;
		boolean oldAPFlag = false;
		if (!deviceManager.isValidAttachmentPoint(sw, port))
			return false;
		AttachmentPoint newAP = new AttachmentPoint(sw, port, lastSeen);
		apList = new ArrayList<AttachmentPoint>();
		if (attachmentPoints != null)
			apList.addAll(attachmentPoints);
		oldAPList = new ArrayList<AttachmentPoint>();
		if (oldAPs != null)
			oldAPList.addAll(oldAPs);
		if (oldAPList.contains(newAP)) {
			int index = oldAPList.indexOf(newAP);
			newAP = oldAPList.remove(index);
			newAP.setLastSeen(lastSeen);
			this.oldAPs = oldAPList;
			oldAPFlag = true;
		}
		Map<DatapathId, AttachmentPoint> apMap = getAPMap(apList);
		if (apMap == null || apMap.isEmpty()) {
			apList.add(newAP);
			attachmentPoints = apList;
			return true;
		}
		DatapathId id = topology.getOpenflowDomainId(sw);
		AttachmentPoint oldAP = apMap.get(id);
			apList = new ArrayList<AttachmentPoint>();
			apList.addAll(apMap.values());
			apList.add(newAP);
			this.attachmentPoints = apList;
		}
		if (oldAP.equals(newAP)) {
			if (newAP.lastSeen.after(oldAP.lastSeen)) {
				oldAP.setLastSeen(newAP.lastSeen);
			}
			this.attachmentPoints = new ArrayList<AttachmentPoint>(
					apMap.values());
		}
		int x = deviceManager.apComparator.compare(oldAP, newAP);
		if (x < 0) {
			apMap.put(id, newAP);
			this.attachmentPoints = new ArrayList<AttachmentPoint>(
					apMap.values());
			oldAPList = new ArrayList<AttachmentPoint>();
			if (oldAPs != null)
				oldAPList.addAll(oldAPs);
			oldAPList.add(oldAP);
			this.oldAPs = oldAPList;
			if (!topology.isInSameBroadcastDomain(oldAP.getSw(),
					oldAP.getPort(), newAP.getSw(), newAP.getPort()))
		} else if (oldAPFlag) {
			oldAPList = new ArrayList<AttachmentPoint>();
			if (oldAPs != null)
				oldAPList.addAll(oldAPs);
			oldAPList.add(newAP);
			this.oldAPs = oldAPList;
		}
		return false;
	}
	public boolean deleteAttachmentPoint(DatapathId sw, OFPort port) {
		AttachmentPoint ap = new AttachmentPoint(sw, port, new Date(0));
		if (this.oldAPs != null) {
			ArrayList<AttachmentPoint> apList = new ArrayList<AttachmentPoint>();
			apList.addAll(this.oldAPs);
			int index = apList.indexOf(ap);
			if (index > 0) {
				apList.remove(index);
				this.oldAPs = apList;
			}
		}
		if (this.attachmentPoints != null) {
			ArrayList<AttachmentPoint> apList = new ArrayList<AttachmentPoint>();
			apList.addAll(this.attachmentPoints);
			int index = apList.indexOf(ap);
			if (index > 0) {
				apList.remove(index);
				this.attachmentPoints = apList;
				return true;
			}
		}
		return false;
	}
	public boolean deleteAttachmentPoint(DatapathId sw) {
		boolean deletedFlag;
		ArrayList<AttachmentPoint> apList;
		ArrayList<AttachmentPoint> modifiedList;
		deletedFlag = false;
		apList = new ArrayList<AttachmentPoint>();
		if (this.oldAPs != null)
			apList.addAll(this.oldAPs);
		modifiedList = new ArrayList<AttachmentPoint>();
		for (AttachmentPoint ap : apList) {
			if (ap.getSw().equals(sw)) {
				deletedFlag = true;
			} else {
				modifiedList.add(ap);
			}
		}
		if (deletedFlag) {
			this.oldAPs = modifiedList;
		}
		deletedFlag = false;
		apList = new ArrayList<AttachmentPoint>();
		if (this.attachmentPoints != null)
			apList.addAll(this.attachmentPoints);
		modifiedList = new ArrayList<AttachmentPoint>();
		for (AttachmentPoint ap : apList) {
			if (ap.getSw().equals(sw)) {
				deletedFlag = true;
			} else {
				modifiedList.add(ap);
			}
		}
		if (deletedFlag) {
			this.attachmentPoints = modifiedList;
			return true;
		}
		return false;
	}
	@Override
	public SwitchPort[] getOldAP() {
		List<SwitchPort> sp = new ArrayList<SwitchPort>();
		SwitchPort[] returnSwitchPorts = new SwitchPort[] {};
		if (oldAPs == null)
			return returnSwitchPorts;
		if (oldAPs.isEmpty())
			return returnSwitchPorts;
		List<AttachmentPoint> oldAPList;
		oldAPList = new ArrayList<AttachmentPoint>();
		if (oldAPs != null)
			oldAPList.addAll(oldAPs);
		removeExpiredAttachmentPoints(oldAPList);
		if (oldAPList != null) {
			for (AttachmentPoint ap : oldAPList) {
				SwitchPort swport = new SwitchPort(ap.getSw(), ap.getPort());
				sp.add(swport);
			}
		}
		return sp.toArray(new SwitchPort[sp.size()]);
	}
	@Override
	public SwitchPort[] getAttachmentPoints() {
		return getAttachmentPoints(false);
	}
	@Override
	public SwitchPort[] getAttachmentPoints(boolean includeError) {
		List<SwitchPort> sp = new ArrayList<SwitchPort>();
		SwitchPort[] returnSwitchPorts = new SwitchPort[] {};
		if (attachmentPoints == null)
			return returnSwitchPorts;
		if (attachmentPoints.isEmpty())
			return returnSwitchPorts;
		List<AttachmentPoint> apList = new ArrayList<AttachmentPoint>(
				attachmentPoints);
		if (apList != null) {
			for (AttachmentPoint ap : apList) {
				SwitchPort swport = new SwitchPort(ap.getSw(), ap.getPort());
				sp.add(swport);
			}
		}
		if (!includeError)
			return sp.toArray(new SwitchPort[sp.size()]);
		List<AttachmentPoint> oldAPList;
		oldAPList = new ArrayList<AttachmentPoint>();
		if (oldAPs != null)
			oldAPList.addAll(oldAPs);
		if (removeExpiredAttachmentPoints(oldAPList))
			this.oldAPs = oldAPList;
		List<AttachmentPoint> dupList;
		Map<DatapathId, AttachmentPoint> apMap = getAPMap(apList);
		dupList = this.getDuplicateAttachmentPoints(oldAPList, apMap);
		if (dupList != null) {
			for (AttachmentPoint ap : dupList) {
				SwitchPort swport = new SwitchPort(ap.getSw(), ap.getPort(),
						ErrorStatus.DUPLICATE_DEVICE);
				sp.add(swport);
			}
		}
		return sp.toArray(new SwitchPort[sp.size()]);
	}
	@Override
	public Long getDeviceKey() {
		return deviceKey;
	}
	@Override
	public MacAddress getMACAddress() {
		return entities[0].getMacAddress();
	}
	@Override
	public String getMACAddressString() {
		return macAddressString;
	}
	@Override
	public VlanVid[] getVlanId() {
		return Arrays.copyOf(vlanIds, vlanIds.length);
	}
	static final EnumSet<DeviceField> ipv4Fields = EnumSet.of(DeviceField.IPv4);
	static final EnumSet<DeviceField> ipv6Fields = EnumSet.of(DeviceField.IPv6);
	@Override
	public IPv4Address[] getIPv4Addresses() {
		TreeSet<IPv4Address> vals = new TreeSet<IPv4Address>();
		for (Entity e : entities) {
			if (e.getIpv4Address().equals(IPv4Address.NONE))
				continue;
			boolean validIP = true;
			Iterator<Device> devices = deviceManager.queryClassByEntity(
					entityClass, ipv4Fields, e);
			while (devices.hasNext()) {
				Device d = devices.next();
				if (deviceKey.equals(d.getDeviceKey()))
					continue;
				for (Entity se : d.entities) {
					if (se.getIpv4Address() != null
							&& se.getIpv4Address().equals(e.getIpv4Address())
							&& !se.getLastSeenTimestamp()
									.equals(Entity.NO_DATE)
							&& 0 < se.getLastSeenTimestamp().compareTo(
									e.getLastSeenTimestamp())) {
						validIP = false;
						break;
					}
				}
				if (!validIP)
					break;
			}
			if (validIP)
				vals.add(e.getIpv4Address());
		}
		return vals.toArray(new IPv4Address[vals.size()]);
	}
	@Override
	public IPv6Address[] getIPv6Addresses() {
		TreeSet<IPv6Address> vals = new TreeSet<IPv6Address>();
		for (Entity e : entities) {
			if (e.getIpv6Address().equals(IPv6Address.NONE))
				continue;
			boolean validIP = true;
			Iterator<Device> devices = deviceManager.queryClassByEntity(
					entityClass, ipv6Fields, e);
			while (devices.hasNext()) {
				Device d = devices.next();
				if (deviceKey.equals(d.getDeviceKey()))
					continue;
				for (Entity se : d.entities) {
					if (se.getIpv6Address() != null
							&& se.getIpv6Address().equals(e.getIpv6Address())
							&& !se.getLastSeenTimestamp()
									.equals(Entity.NO_DATE)
							&& 0 < se.getLastSeenTimestamp().compareTo(
									e.getLastSeenTimestamp())) {
						validIP = false;
						break;
					}
				}
				if (!validIP)
					break;
			}
			if (validIP)
				vals.add(e.getIpv6Address());
		}
		return vals.toArray(new IPv6Address[vals.size()]);
	}
	@Override
	public VlanVid[] getSwitchPortVlanIds(SwitchPort swp) {
		TreeSet<VlanVid> vals = new TreeSet<VlanVid>();
		for (Entity e : entities) {
			if (e.switchDPID.equals(swp.getSwitchDPID())
					&& e.switchPort.equals(swp.getPort())) {
				if (e.getVlan() == null)
					vals.add(VlanVid.ZERO);
				else
					vals.add(e.getVlan());
			}
		}
		return vals.toArray(new VlanVid[vals.size()]);
	}
	@Override
	public Date getLastSeen() {
		Date d = null;
		for (int i = 0; i < entities.length; i++) {
			if (d == null
					|| entities[i].getLastSeenTimestamp().compareTo(d) > 0)
				d = entities[i].getLastSeenTimestamp();
		}
		return d;
	}
	@Override
	public IEntityClass getEntityClass() {
		return entityClass;
	}
	public Entity[] getEntities() {
		return entities;
	}
	public String getDHCPClientName() {
		return dhcpClientName;
	}
	protected int entityIndex(Entity entity) {
		return Arrays.binarySearch(entities, entity);
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
				+ ((deviceKey == null) ? 0 : deviceKey.hashCode());
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
		Device other = (Device) obj;
		if (deviceKey == null) {
			if (other.deviceKey != null)
				return false;
		} else if (!deviceKey.equals(other.deviceKey))
			return false;
		if (!Arrays.equals(entities, other.entities))
			return false;
		return true;
	}
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Device [deviceKey=");
		builder.append(deviceKey);
		builder.append(", entityClass=");
		builder.append(entityClass.getName());
		builder.append(", MAC=");
		builder.append(macAddressString);
		builder.append(", IPv4s=[");
		boolean isFirst = true;
		for (IPv4Address ip : getIPv4Addresses()) {
			if (!isFirst)
				builder.append(", ");
			isFirst = false;
			builder.append(ip.toString());
		}
		builder.append("], IPv6s=[");
		isFirst = true;
		for (IPv6Address ip : getIPv6Addresses()) {
			if (!isFirst)
				builder.append(", ");
			isFirst = false;
			builder.append(ip.toString());
		}
		builder.append("], APs=");
		builder.append(Arrays.toString(getAttachmentPoints(true)));
		builder.append("]");
		return builder.toString();
	}
}
