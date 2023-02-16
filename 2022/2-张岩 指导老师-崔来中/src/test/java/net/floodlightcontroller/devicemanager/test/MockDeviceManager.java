package net.floodlightcontroller.devicemanager.test;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
import org.sdnplatform.sync.test.MockSyncService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.IDeviceListener;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.IEntityClassifierService;
import net.floodlightcontroller.devicemanager.internal.AttachmentPoint;
import net.floodlightcontroller.devicemanager.internal.Device;
import net.floodlightcontroller.devicemanager.internal.DeviceManagerImpl;
import net.floodlightcontroller.devicemanager.internal.Entity;
public class MockDeviceManager extends DeviceManagerImpl {
	public void setEntityClassifier(IEntityClassifierService ecs) {
		this.entityClassifier = ecs;
		try {
			this.startUp(null);
		} catch (FloodlightModuleException e) {
			throw new RuntimeException(e);
		}
	}
	public IDevice learnEntity(MacAddress macAddress, VlanVid vlan,
			IPv4Address ipv4Address, IPv6Address ipv6Address, DatapathId switchDPID,
			OFPort switchPort,
			boolean processUpdates) {
		List<IDeviceListener> listeners = deviceListeners.getOrderedListeners();
		if (!processUpdates) {
			deviceListeners.clearListeners();
		}
		IDevice res =  learnDeviceByEntity(new Entity(macAddress, 
				vlan, ipv4Address, ipv6Address, switchDPID, switchPort, new Date()));
		if (listeners != null) {
			for (IDeviceListener listener : listeners) {
				deviceListeners.addListener("device", listener);
			}
		}
		return res;
	}
	@Override 
	public void deleteDevice(Device device) {
		super.deleteDevice(device);
	}
	public IDevice learnEntity(MacAddress macAddress, VlanVid vlan,
			IPv4Address ipv4Address, IPv6Address ipv6Address, DatapathId switchDPID,
			OFPort switchPort) {
		return learnEntity(macAddress, vlan, ipv4Address, ipv6Address, switchDPID, switchPort, true);
	}
	@Override
	protected Device allocateDevice(Long deviceKey,
			Entity entity,
			IEntityClass entityClass) {
		return new MockDevice(this, deviceKey, entity, entityClass);
	}
	@Override
	protected Device allocateDevice(Long deviceKey,
			String dhcpClientName,
			List<AttachmentPoint> aps,
			List<AttachmentPoint> trueAPs,
			Collection<Entity> entities,
			IEntityClass entityClass) {
		return new MockDevice(this, deviceKey, aps, trueAPs, entities, entityClass);
	}
	@Override
	protected Device allocateDevice(Device device,
			Entity entity,
			int insertionpoint) {
		return new MockDevice(device, entity, insertionpoint);
	}
	@Override
	public void init(FloodlightModuleContext fmc) throws FloodlightModuleException {
		super.init(fmc);
		setSyncServiceIfNotSet(new MockSyncService());
	}
}