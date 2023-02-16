package net.floodlightcontroller.devicemanager.internal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nonnull;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.HAListenerTypeMarker;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IInfoProvider;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.ListenerDispatcher;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.debugcounter.IDebugCounter;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugevent.DebugEventService.EventCategoryBuilder;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.debugevent.IDebugEventService.EventColumn;
import net.floodlightcontroller.debugevent.IDebugEventService.EventFieldType;
import net.floodlightcontroller.debugevent.IDebugEventService.EventType;
import net.floodlightcontroller.debugevent.IEventCategory;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.IEntityClassListener;
import net.floodlightcontroller.devicemanager.IEntityClassifierService;
import net.floodlightcontroller.devicemanager.IDeviceListener;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.devicemanager.internal.DeviceSyncRepresentation.SyncEntity;
import net.floodlightcontroller.devicemanager.web.DeviceRoutable;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LDUpdate;
import net.floodlightcontroller.packet.ARP;
import net.floodlightcontroller.packet.DHCP;
import net.floodlightcontroller.packet.DHCPOption;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.IPv6;
import net.floodlightcontroller.packet.UDP;
import net.floodlightcontroller.packet.DHCP.DHCPOptionCode;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import net.floodlightcontroller.topology.ITopologyListener;
import net.floodlightcontroller.topology.ITopologyService;
import net.floodlightcontroller.util.MultiIterator;
import static net.floodlightcontroller.devicemanager.internal.
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.VlanVid;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.sdnplatform.sync.IClosableIterator;
import org.sdnplatform.sync.IStoreClient;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.ISyncService.Scope;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.error.ObsoleteVersionException;
import org.sdnplatform.sync.error.SyncException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class DeviceManagerImpl implements IDeviceService, IOFMessageListener, ITopologyListener, IFloodlightModule, IEntityClassListener, IInfoProvider {
	protected static Logger logger = LoggerFactory.getLogger(DeviceManagerImpl.class);
	protected IFloodlightProviderService floodlightProvider;
	protected ITopologyService topology;
	protected IStorageSourceService storageSource;
	protected IRestApiService restApi;
	protected IThreadPoolService threadPool;
	protected IDebugCounterService debugCounters;
	private ISyncService syncService;
	private IStoreClient<String, DeviceSyncRepresentation> storeClient;
	private DeviceSyncManager deviceSyncManager;
	public static final String MODULE_NAME = "devicemanager";
	public static final String PACKAGE = DeviceManagerImpl.class.getPackage().getName();
	public IDebugCounter cntIncoming;
	public IDebugCounter cntReconcileRequest;
	public IDebugCounter cntReconcileNoSource;
	public IDebugCounter cntReconcileNoDest;
	public IDebugCounter cntInvalidSource;
	public IDebugCounter cntInvalidDest;
	public IDebugCounter cntNoSource;
	public IDebugCounter cntNoDest;
	public IDebugCounter cntDhcpClientNameSnooped;
	public IDebugCounter cntDeviceOnInternalPortNotLearned;
	public IDebugCounter cntPacketNotAllowed;
	public IDebugCounter cntNewDevice;
	public IDebugCounter cntPacketOnInternalPortForKnownDevice;
	public IDebugCounter cntNewEntity;
	public IDebugCounter cntDeviceChanged;
	public IDebugCounter cntDeviceMoved;
	public IDebugCounter cntCleanupEntitiesRuns;
	public IDebugCounter cntEntityRemovedTimeout;
	public IDebugCounter cntDeviceDeleted;
	public IDebugCounter cntDeviceReclassifyDelete;
	public IDebugCounter cntDeviceStrored;
	public IDebugCounter cntDeviceStoreThrottled;
	public IDebugCounter cntDeviceRemovedFromStore;
	public IDebugCounter cntSyncException;
	public IDebugCounter cntDevicesFromStore;
	public IDebugCounter cntConsolidateStoreRuns;
	public IDebugCounter cntConsolidateStoreDevicesRemoved;
	public IDebugCounter cntTransitionToMaster;
	private IDebugEventService debugEventService;
	private IEventCategory<DeviceEvent> debugEventCategory;
	private boolean isMaster = false;
	static final String DEVICE_SYNC_STORE_NAME =
			DeviceManagerImpl.class.getCanonicalName() + ".stateStore";
	private int syncStoreWriteIntervalMs = DEFAULT_SYNC_STORE_WRITE_INTERVAL_MS;
	static final int DEFAULT_INITIAL_SYNC_STORE_CONSOLIDATE_MS =
	private int initialSyncStoreConsolidateMs =
			DEFAULT_INITIAL_SYNC_STORE_CONSOLIDATE_MS;
	static final int DEFAULT_SYNC_STORE_CONSOLIDATE_INTERVAL_MS =
	private final int syncStoreConsolidateIntervalMs =
			DEFAULT_SYNC_STORE_CONSOLIDATE_INTERVAL_MS;
	protected ConcurrentHashMap<Long, Device> deviceMap;
	protected AtomicLong deviceKeyCounter = new AtomicLong(0);
	protected DeviceUniqueIndex primaryIndex;
	protected Map<EnumSet<DeviceField>, DeviceIndex> secondaryIndexMap;
	protected ConcurrentHashMap<String, ClassState> classStateMap;
	protected Set<EnumSet<DeviceField>> perClassIndices;
	protected IEntityClassifierService entityClassifier;
	protected class ClassState {
		protected DeviceUniqueIndex classIndex;
		protected Map<EnumSet<DeviceField>, DeviceIndex> secondaryIndexMap;
		public ClassState(IEntityClass clazz) {
			EnumSet<DeviceField> keyFields = clazz.getKeyFields();
			EnumSet<DeviceField> primaryKeyFields =
					entityClassifier.getKeyFields();
			boolean keyFieldsMatchPrimary =
					primaryKeyFields.equals(keyFields);
			if (!keyFieldsMatchPrimary)
				classIndex = new DeviceUniqueIndex(keyFields);
			secondaryIndexMap =
					new HashMap<EnumSet<DeviceField>, DeviceIndex>();
			for (EnumSet<DeviceField> fields : perClassIndices) {
				secondaryIndexMap.put(fields,
						new DeviceMultiIndex(fields));
			}
		}
	}
	protected ListenerDispatcher<String,IDeviceListener> deviceListeners;
	protected static class DeviceUpdate {
		public enum Change {
			ADD, DELETE, CHANGE;
		}
		protected Device device;
		protected Change change;
		protected EnumSet<DeviceField> fieldsChanged;
		public DeviceUpdate(Device device, Change change,
				EnumSet<DeviceField> fieldsChanged) {
			super();
			this.device = device;
			this.change = change;
			this.fieldsChanged = fieldsChanged;
		}
		@Override
		public String toString() {
			String devIdStr = device.getEntityClass().getName() + "::" +
					device.getMACAddressString();
			return "DeviceUpdate [device=" + devIdStr + ", change=" + change
					+ ", fieldsChanged=" + fieldsChanged + "]";
		}
	}
	protected class AttachmentPointComparator
	implements Comparator<AttachmentPoint> {
		public AttachmentPointComparator() {
			super();
		}
		@Override
		public int compare(AttachmentPoint oldAP, AttachmentPoint newAP) {
			DatapathId oldSw = oldAP.getSw();
			OFPort oldPort = oldAP.getPort();
			DatapathId oldDomain = topology.getOpenflowDomainId(oldSw);
			boolean oldBD = topology.isBroadcastDomainPort(oldSw, oldPort);
			DatapathId newSw = newAP.getSw();
			OFPort newPort = newAP.getPort();
			DatapathId newDomain = topology.getOpenflowDomainId(newSw);
			boolean newBD = topology.isBroadcastDomainPort(newSw, newPort);
			if (oldDomain.getLong() < newDomain.getLong()) return -1;
			else if (oldDomain.getLong() > newDomain.getLong()) return 1;
			if (oldPort != OFPort.LOCAL &&
					newPort == OFPort.LOCAL) {
				return -1;
			} else if (oldPort == OFPort.LOCAL &&
					newPort != OFPort.LOCAL) {
				return 1;
			}
				return -compare(newAP, oldAP);
			long activeOffset = 0;
			if (!topology.isConsistent(oldSw, oldPort, newSw, newPort)) {
				if (!newBD && oldBD) {
					return -1;
				}
				if (newBD && oldBD) {
					activeOffset = AttachmentPoint.EXTERNAL_TO_EXTERNAL_TIMEOUT;
				}
				else if (newBD && !oldBD){
					activeOffset = AttachmentPoint.OPENFLOW_TO_EXTERNAL_TIMEOUT;
				}
			} else {
				activeOffset = AttachmentPoint.CONSISTENT_TIMEOUT;
			}
			if ((newAP.getActiveSince().getTime() > oldAP.getLastSeen().getTime() + activeOffset) ||
					(newAP.getLastSeen().getTime() > oldAP.getLastSeen().getTime() +
							AttachmentPoint.INACTIVITY_INTERVAL)) {
				return -1;
			}
			return 1;
		}
	}
	public AttachmentPointComparator apComparator;
	private Set<SwitchPort> suppressAPs;
	public SingletonTask entityCleanupTask;
	private SingletonTask storeConsolidateTask;
	protected HAListenerDelegate haListenerDelegate;
	@Override
	public IDevice getDevice(Long deviceKey) {
		return deviceMap.get(deviceKey);
	}
	@Override
	public IDevice findDevice(@Nonnull MacAddress macAddress, VlanVid vlan,
			@Nonnull IPv4Address ipv4Address, @Nonnull IPv6Address ipv6Address,
			@Nonnull DatapathId switchDPID, @Nonnull OFPort switchPort)
					throws IllegalArgumentException {
		if (macAddress == null) {
    		throw new IllegalArgumentException("MAC address cannot be null. Try MacAddress.NONE if intention is 'no MAC'");
    	}
    	if (ipv4Address == null) {
    		throw new IllegalArgumentException("IPv4 address cannot be null. Try IPv4Address.NONE if intention is 'no IPv4'");
    	}
    	if (ipv6Address == null) {
    		throw new IllegalArgumentException("IPv6 address cannot be null. Try IPv6Address.NONE if intention is 'no IPv6'");
    	}
    	if (vlan == null) {
    		throw new IllegalArgumentException("VLAN cannot be null. Try VlanVid.ZERO if intention is 'no VLAN / untagged'");
    	}
    	if (switchDPID == null) {
    		throw new IllegalArgumentException("Switch DPID cannot be null. Try DatapathId.NONE if intention is 'no DPID'");
    	}
    	if (switchPort == null) {
    		throw new IllegalArgumentException("Switch port cannot be null. Try OFPort.ZERO if intention is 'no port'");
    	}
		Entity e = new Entity(macAddress, vlan, 
				ipv4Address, ipv6Address, 
				switchDPID, switchPort, Entity.NO_DATE);
		if (!allKeyFieldsPresent(e, entityClassifier.getKeyFields())) {
			throw new IllegalArgumentException("Not all key fields specified."
					+ " Required fields: " + entityClassifier.getKeyFields());
		}
		return findDeviceByEntity(e);
	}
	@Override
	public IDevice findClassDevice(@Nonnull IEntityClass entityClass, @Nonnull MacAddress macAddress,
			@Nonnull VlanVid vlan, @Nonnull IPv4Address ipv4Address, @Nonnull IPv6Address ipv6Address)
					throws IllegalArgumentException {
		if (entityClass == null) {
    		throw new IllegalArgumentException("Entity class cannot be null.");
    	}
		if (macAddress == null) {
    		throw new IllegalArgumentException("MAC address cannot be null. Try MacAddress.NONE if intention is 'no MAC'");
    	}
    	if (ipv4Address == null) {
    		throw new IllegalArgumentException("IPv4 address cannot be null. Try IPv4Address.NONE if intention is 'no IPv4'");
    	}
    	if (ipv6Address == null) {
    		throw new IllegalArgumentException("IPv6 address cannot be null. Try IPv6Address.NONE if intention is 'no IPv6'");
    	}
    	if (vlan == null) {
    		throw new IllegalArgumentException("VLAN cannot be null. Try VlanVid.ZERO if intention is 'no VLAN / untagged'");
    	}
		Entity e = new Entity(macAddress, vlan, ipv4Address, ipv6Address, DatapathId.NONE, OFPort.ZERO, Entity.NO_DATE);
		if (!allKeyFieldsPresent(e, entityClass.getKeyFields())) {
			throw new IllegalArgumentException("Not all key fields and/or "
					+ " no source device specified. Required fields: " +
					entityClassifier.getKeyFields());
		}
		return findDestByEntity(entityClass, e);
	}
	@Override
	public Collection<? extends IDevice> getAllDevices() {
		return Collections.unmodifiableCollection(deviceMap.values());
	}
	@Override
	public void addIndex(boolean perClass,
			EnumSet<DeviceField> keyFields) {
		if (perClass) {
			perClassIndices.add(keyFields);
		} else {
			secondaryIndexMap.put(keyFields,
					new DeviceMultiIndex(keyFields));
		}
	}
	@Override
	public Iterator<? extends IDevice> queryDevices(@Nonnull MacAddress macAddress,
			VlanVid vlan,
			@Nonnull IPv4Address ipv4Address,
			@Nonnull IPv6Address ipv6Address,
			@Nonnull DatapathId switchDPID,
			@Nonnull OFPort switchPort) {
		if (macAddress == null) {
    		throw new IllegalArgumentException("MAC address cannot be null. Try MacAddress.NONE if intention is 'no MAC'");
    	}
    	if (ipv4Address == null) {
    		throw new IllegalArgumentException("IPv4 address cannot be null. Try IPv4Address.NONE if intention is 'no IPv4'");
    	}
    	if (ipv6Address == null) {
    		throw new IllegalArgumentException("IPv6 address cannot be null. Try IPv6Address.NONE if intention is 'no IPv6'");
    	}
    	if (switchDPID == null) {
    		throw new IllegalArgumentException("Switch DPID cannot be null. Try DatapathId.NONE if intention is 'no DPID'");
    	}
    	if (switchPort == null) {
    		throw new IllegalArgumentException("Switch port cannot be null. Try OFPort.ZERO if intention is 'no port'");
    	}
		DeviceIndex index = null;
		if (secondaryIndexMap.size() > 0) {
			EnumSet<DeviceField> keys =
					getEntityKeys(macAddress, vlan, ipv4Address, ipv6Address,
							switchDPID, switchPort);
			index = secondaryIndexMap.get(keys);
		}
		Iterator<Device> deviceIterator = null;
		if (index == null) {
			deviceIterator = deviceMap.values().iterator();
		} else {
			Entity entity = new Entity(macAddress,
					vlan,
					ipv4Address,
					ipv6Address,
					switchDPID,
					switchPort,
					Entity.NO_DATE);
			deviceIterator =
					new DeviceIndexInterator(this, index.queryByEntity(entity));
		}
		DeviceIterator di =
				new DeviceIterator(deviceIterator,
						null,
						macAddress,
						vlan,
						ipv4Address,
						ipv6Address,
						switchDPID,
						switchPort);
		return di;
	}
	@Override
	public Iterator<? extends IDevice> queryClassDevices(@Nonnull IEntityClass entityClass,
			@Nonnull MacAddress macAddress,
			@Nonnull VlanVid vlan,
			@Nonnull IPv4Address ipv4Address,
			@Nonnull IPv6Address ipv6Address,
			@Nonnull DatapathId switchDPID,
			@Nonnull OFPort switchPort) {
		if (macAddress == null) {
    		throw new IllegalArgumentException("MAC address cannot be null. Try MacAddress.NONE if intention is 'no MAC'");
    	}
    	if (ipv4Address == null) {
    		throw new IllegalArgumentException("IPv4 address cannot be null. Try IPv4Address.NONE if intention is 'no IPv4'");
    	}
    	if (ipv6Address == null) {
    		throw new IllegalArgumentException("IPv6 address cannot be null. Try IPv6Address.NONE if intention is 'no IPv6'");
    	}
    	if (switchDPID == null) {
    		throw new IllegalArgumentException("Switch DPID cannot be null. Try DatapathId.NONE if intention is 'no DPID'");
    	}
    	if (switchPort == null) {
    		throw new IllegalArgumentException("Switch port cannot be null. Try OFPort.ZERO if intention is 'no port'");
    	}
		ArrayList<Iterator<Device>> iterators =
				new ArrayList<Iterator<Device>>();
		ClassState classState = getClassState(entityClass);
		DeviceIndex index = null;
		if (classState.secondaryIndexMap.size() > 0) {
			EnumSet<DeviceField> keys =
					getEntityKeys(macAddress, vlan, ipv4Address,
							ipv6Address, switchDPID, switchPort);
			index = classState.secondaryIndexMap.get(keys);
		}
		Iterator<Device> iter;
		if (index == null) {
			index = classState.classIndex;
			if (index == null) {
				return new DeviceIterator(deviceMap.values().iterator(),
						new IEntityClass[] { entityClass },
						macAddress, vlan, ipv4Address,
						ipv6Address, switchDPID, switchPort);
			} else {
				iter = new DeviceIndexInterator(this, index.getAll());
			}
		} else {
			Entity entity =
					new Entity(macAddress,
							vlan,
							ipv4Address,
							ipv6Address,
							switchDPID,
							switchPort,
							Entity.NO_DATE);
			iter = new DeviceIndexInterator(this,
					index.queryByEntity(entity));
		}
		iterators.add(iter);
		return new MultiIterator<Device>(iterators.iterator());
	}
	protected Iterator<Device> getDeviceIteratorForQuery(@Nonnull MacAddress macAddress,
			VlanVid vlan,
			@Nonnull IPv4Address ipv4Address,
			@Nonnull IPv6Address ipv6Address,
			@Nonnull DatapathId switchDPID,
			@Nonnull OFPort switchPort) {
		if (macAddress == null) {
    		throw new IllegalArgumentException("MAC address cannot be null. Try MacAddress.NONE if intention is 'no MAC'");
    	}
    	if (ipv4Address == null) {
    		throw new IllegalArgumentException("IPv4 address cannot be null. Try IPv4Address.NONE if intention is 'no IPv4'");
    	}
    	if (ipv6Address == null) {
    		throw new IllegalArgumentException("IPv6 address cannot be null. Try IPv6Address.NONE if intention is 'no IPv6'");
    	}
    	if (switchDPID == null) {
    		throw new IllegalArgumentException("Switch DPID cannot be null. Try DatapathId.NONE if intention is 'no DPID'");
    	}
    	if (switchPort == null) {
    		throw new IllegalArgumentException("Switch port cannot be null. Try OFPort.ZERO if intention is 'no port'");
    	}
		DeviceIndex index = null;
		if (secondaryIndexMap.size() > 0) {
			EnumSet<DeviceField> keys =
					getEntityKeys(macAddress, vlan, ipv4Address,
							ipv6Address, switchDPID, switchPort);
			index = secondaryIndexMap.get(keys);
		}
		Iterator<Device> deviceIterator = null;
		if (index == null) {
			deviceIterator = deviceMap.values().iterator();
		} else {
			Entity entity = new Entity(macAddress,
					vlan,
					ipv4Address,
					ipv6Address,
					switchDPID,
					switchPort,
					Entity.NO_DATE);
			deviceIterator =
					new DeviceIndexInterator(this, index.queryByEntity(entity));
		}
		DeviceIterator di =
				new DeviceIterator(deviceIterator,
						null,
						macAddress,
						vlan,
						ipv4Address,
						ipv6Address,
						switchDPID,
						switchPort);
		return di;
	}
	@Override
	public void addListener(IDeviceListener listener) {
		deviceListeners.addListener("device", listener);
		logListeners();
	}
	@Override
	public void addSuppressAPs(DatapathId swId, OFPort port) {
		suppressAPs.add(new SwitchPort(swId, port));
	}
	@Override
	public void removeSuppressAPs(DatapathId swId, OFPort port) {
		suppressAPs.remove(new SwitchPort(swId, port));
	}
	@Override
	public Set<SwitchPort> getSuppressAPs() {
		return Collections.unmodifiableSet(suppressAPs);
	}
	private void logListeners() {
		List<IDeviceListener> listeners = deviceListeners.getOrderedListeners();
		if (listeners != null) {
			StringBuffer sb = new StringBuffer();
			sb.append("DeviceListeners: ");
			for (IDeviceListener l : listeners) {
				sb.append(l.getName());
				sb.append(",");
			}
			logger.debug(sb.toString());
		}
	}
	private class DeviceDebugEventLogger implements IDeviceListener {
		@Override
		public String getName() {
			return "deviceDebugEventLogger";
		}
		@Override
		public boolean isCallbackOrderingPrereq(String type, String name) {
			return false;
		}
		@Override
		public boolean isCallbackOrderingPostreq(String type, String name) {
			return false;
		}
		@Override
		public void deviceAdded(IDevice device) {
			generateDeviceEvent(device, "host-added");
		}
		@Override
		public void deviceRemoved(IDevice device) {
			generateDeviceEvent(device, "host-removed");
		}
		@Override
		public void deviceMoved(IDevice device) {
			generateDeviceEvent(device, "host-moved");
		}
		@Override
		public void deviceIPV4AddrChanged(IDevice device) {
			generateDeviceEvent(device, "host-ipv4-addr-changed");
		}
		@Override
		public void deviceIPV6AddrChanged(IDevice device) {
			generateDeviceEvent(device, "host-ipv6-addr-changed");
		}
		@Override
		public void deviceVlanChanged(IDevice device) {
			generateDeviceEvent(device, "host-vlan-changed");
		}
		private void generateDeviceEvent(IDevice device, String reason) {
			List<IPv4Address> ipv4Addresses =
					new ArrayList<IPv4Address>(Arrays.asList(device.getIPv4Addresses()));
			List<IPv6Address> ipv6Addresses =
					new ArrayList<IPv6Address>(Arrays.asList(device.getIPv6Addresses()));
			List<SwitchPort> oldAps =
					new ArrayList<SwitchPort>(Arrays.asList(device.getOldAP()));
			List<SwitchPort> currentAps =
					new ArrayList<SwitchPort>(Arrays.asList(device.getAttachmentPoints()));
			List<VlanVid> vlanIds =
					new ArrayList<VlanVid>(Arrays.asList(device.getVlanId()));
			debugEventCategory.newEventNoFlush(new DeviceEvent(device.getMACAddress(),
					ipv4Addresses,
					ipv6Addresses,
					oldAps,
					currentAps,
					vlanIds, reason));
		}
	}
	@Override
	public Map<String, Object> getInfo(String type) {
		if (!"summary".equals(type))
			return null;
		Map<String, Object> info = new HashMap<String, Object>();
		info.put("# hosts", deviceMap.size());
		return info;
	}
	@Override
	public String getName() {
		return MODULE_NAME;
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return ((type == OFType.PACKET_IN || type == OFType.FLOW_MOD)
				&& name.equals("topology"));
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		return false;
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg,
			FloodlightContext cntx) {
		switch (msg.getType()) {
		case PACKET_IN:
			cntIncoming.increment();
			return this.processPacketInMessage(sw, (OFPacketIn) msg, cntx);
		default:
			break;
		}
		return Command.CONTINUE;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IDeviceService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService>
	getServiceImpls() {
		Map<Class<? extends IFloodlightService>,
		IFloodlightService> m =
		new HashMap<Class<? extends IFloodlightService>,
		IFloodlightService>();
		m.put(IDeviceService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IStorageSourceService.class);
		l.add(ITopologyService.class);
		l.add(IRestApiService.class);
		l.add(IThreadPoolService.class);
		l.add(IEntityClassifierService.class);
		l.add(ISyncService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext fmc) throws FloodlightModuleException {
		this.perClassIndices =
				new HashSet<EnumSet<DeviceField>>();
		addIndex(true, EnumSet.of(DeviceField.IPv4));
		addIndex(true, EnumSet.of(DeviceField.IPv6));
		this.deviceListeners = new ListenerDispatcher<String, IDeviceListener>();
		this.suppressAPs = Collections.newSetFromMap(
				new ConcurrentHashMap<SwitchPort, Boolean>());
		this.floodlightProvider =
				fmc.getServiceImpl(IFloodlightProviderService.class);
		this.storageSource =
				fmc.getServiceImpl(IStorageSourceService.class);
		this.topology =
				fmc.getServiceImpl(ITopologyService.class);
		this.restApi = fmc.getServiceImpl(IRestApiService.class);
		this.threadPool = fmc.getServiceImpl(IThreadPoolService.class);
		this.entityClassifier = fmc.getServiceImpl(IEntityClassifierService.class);
		this.debugCounters = fmc.getServiceImpl(IDebugCounterService.class);
		this.debugEventService = fmc.getServiceImpl(IDebugEventService.class);
		this.syncService = fmc.getServiceImpl(ISyncService.class);
		this.deviceSyncManager = new DeviceSyncManager();
		this.haListenerDelegate = new HAListenerDelegate();
		registerDeviceManagerDebugCounters();
		registerDeviceManagerDebugEvents();
		this.addListener(new DeviceDebugEventLogger());
	}
	private void registerDeviceManagerDebugEvents() throws FloodlightModuleException {
		if (debugEventService == null) {
			logger.error("debugEventService should not be null");
		}
		EventCategoryBuilder<DeviceEvent> ecb = debugEventService.buildEvent(DeviceEvent.class);
		debugEventCategory = ecb.setModuleName(PACKAGE)
				.setEventName("hostevent")
				.setEventDescription("Host added, removed, updated, or moved")
				.setEventType(EventType.ALWAYS_LOG)
				.setBufferCapacity(500)
				.setAckable(false)
				.register();
	}
	@Override
	public void startUp(FloodlightModuleContext fmc)
			throws FloodlightModuleException {
		isMaster = (floodlightProvider.getRole() == HARole.ACTIVE);
		primaryIndex = new DeviceUniqueIndex(entityClassifier.getKeyFields());
		secondaryIndexMap = new HashMap<EnumSet<DeviceField>, DeviceIndex>();
		deviceMap = new ConcurrentHashMap<Long, Device>();
		classStateMap =
				new ConcurrentHashMap<String, ClassState>();
		apComparator = new AttachmentPointComparator();
		floodlightProvider.addOFMessageListener(OFType.PACKET_IN, this);
		floodlightProvider.addHAListener(this.haListenerDelegate);
		if (topology != null)
			topology.addListener(this);
		entityClassifier.addListener(this);
		ScheduledExecutorService ses = threadPool.getScheduledExecutor();
		Runnable ecr = new Runnable() {
			@Override
			public void run() {
				cleanupEntities();
				entityCleanupTask.reschedule(ENTITY_CLEANUP_INTERVAL,
						TimeUnit.SECONDS);
			}
		};
		entityCleanupTask = new SingletonTask(ses, ecr);
		entityCleanupTask.reschedule(ENTITY_CLEANUP_INTERVAL,
				TimeUnit.SECONDS);
		Runnable consolidateStoreRunner = new Runnable() {
			@Override
			public void run() {
				deviceSyncManager.consolidateStore();
				storeConsolidateTask.reschedule(syncStoreConsolidateIntervalMs,
						TimeUnit.MILLISECONDS);
			}
		};
		storeConsolidateTask = new SingletonTask(ses, consolidateStoreRunner);
		if (isMaster)
			storeConsolidateTask.reschedule(syncStoreConsolidateIntervalMs,
					TimeUnit.MILLISECONDS);
		if (restApi != null) {
			restApi.addRestletRoutable(new DeviceRoutable());
		} else {
			logger.debug("Could not instantiate REST API");
		}
		try {
			this.syncService.registerStore(DEVICE_SYNC_STORE_NAME, Scope.LOCAL);
			this.storeClient = this.syncService
					.getStoreClient(DEVICE_SYNC_STORE_NAME,
							String.class,
							DeviceSyncRepresentation.class);
		} catch (SyncException e) {
			throw new FloodlightModuleException("Error while setting up sync service", e);
		}
		floodlightProvider.addInfoProvider("summary", this);
	}
	private void registerDeviceManagerDebugCounters() throws FloodlightModuleException {
		if (debugCounters == null) {
			logger.error("Debug Counter Service not found.");
		}
		debugCounters.registerModule(PACKAGE);
		cntIncoming = debugCounters.registerCounter(PACKAGE, "incoming",
				"All incoming packets seen by this module");
		cntReconcileRequest = debugCounters.registerCounter(PACKAGE,
				"reconcile-request",
				"Number of flows that have been received for reconciliation by " +
				"this module");
		cntReconcileNoSource = debugCounters.registerCounter(PACKAGE,
				"reconcile-no-source-device",
				"Number of flow reconcile events that failed because no source " +
		cntReconcileNoDest = debugCounters.registerCounter(PACKAGE,
				"reconcile-no-dest-device",
				"Number of flow reconcile events that failed because no " +
		cntInvalidSource = debugCounters.registerCounter(PACKAGE,
				"invalid-source",
				"Number of packetIns that were discarded because the source " +
						"MAC was invalid (broadcast, multicast, or zero)", IDebugCounterService.MetaData.WARN);
		cntNoSource = debugCounters.registerCounter(PACKAGE, "no-source-device",
				"Number of packetIns that were discarded because the " +
						"could not identify a source device. This can happen if a " +
						"packet is not allowed, appears on an illegal port, does not " +
						"have a valid address space, etc.", IDebugCounterService.MetaData.WARN);
		cntInvalidDest = debugCounters.registerCounter(PACKAGE,
				"invalid-dest",
				"Number of packetIns that were discarded because the dest " +
						"MAC was invalid (zero)", IDebugCounterService.MetaData.WARN);
		cntNoDest = debugCounters.registerCounter(PACKAGE, "no-dest-device",
				"Number of packetIns that did not have an associated " +
						"destination device. E.g., because the destination MAC is " +
				"broadcast/multicast or is not yet known to the controller.");
		cntDhcpClientNameSnooped = debugCounters.registerCounter(PACKAGE,
				"dhcp-client-name-snooped",
				"Number of times a DHCP client name was snooped from a " +
				"packetIn.");
		cntDeviceOnInternalPortNotLearned = debugCounters.registerCounter(
				PACKAGE,
				"device-on-internal-port-not-learned",
				"Number of times packetIn was received on an internal port and" +
						"no source device is known for the source MAC. The packetIn is " +
						"discarded.", IDebugCounterService.MetaData.WARN);
		cntPacketNotAllowed = debugCounters.registerCounter(PACKAGE,
				"packet-not-allowed",
				"Number of times a packetIn was not allowed due to spoofing " +
		cntNewDevice = debugCounters.registerCounter(PACKAGE, "new-device",
				"Number of times a new device was learned");
		cntPacketOnInternalPortForKnownDevice = debugCounters.registerCounter(
				PACKAGE,
				"packet-on-internal-port-for-known-device",
				"Number of times a packetIn was received on an internal port " +
				"for a known device.");
		cntNewEntity = debugCounters.registerCounter(PACKAGE, "new-entity",
				"Number of times a new entity was learned for an existing device");
		cntDeviceChanged = debugCounters.registerCounter(PACKAGE, "device-changed",
				"Number of times device properties have changed");
		cntDeviceMoved = debugCounters.registerCounter(PACKAGE, "device-moved",
				"Number of times devices have moved");
		cntCleanupEntitiesRuns = debugCounters.registerCounter(PACKAGE,
				"cleanup-entities-runs",
				"Number of times the entity cleanup task has been run");
		cntEntityRemovedTimeout = debugCounters.registerCounter(PACKAGE,
				"entity-removed-timeout",
				"Number of times entities have been removed due to timeout " +
						"(entity has been inactive for " + ENTITY_TIMEOUT/1000 + "s)");
		cntDeviceDeleted = debugCounters.registerCounter(PACKAGE, "device-deleted",
				"Number of devices that have been removed due to inactivity");
		cntDeviceReclassifyDelete = debugCounters.registerCounter(PACKAGE,
				"device-reclassify-delete",
				"Number of devices that required reclassification and have been " +
				"temporarily delete for reclassification");
		cntDeviceStrored = debugCounters.registerCounter(PACKAGE, "device-stored",
				"Number of device entries written or updated to the sync store");
		cntDeviceStoreThrottled = debugCounters.registerCounter(PACKAGE,
				"device-store-throttled",
				"Number of times a device update to the sync store was " +
						"requested but not performed because the same device entities " +
				"have recently been updated already");
		cntDeviceRemovedFromStore = debugCounters.registerCounter(PACKAGE,
				"device-removed-from-store",
				"Number of devices that were removed from the sync store " +
						"because the local controller removed the device due to " +
				"inactivity");
		cntSyncException = debugCounters.registerCounter(PACKAGE, "sync-exception",
				"Number of times an operation on the sync store resulted in " +
		cntDevicesFromStore = debugCounters.registerCounter(PACKAGE,
				"devices-from-store",
				"Number of devices that were read from the sync store after " +
				"the local controller transitioned from SLAVE to MASTER");
		cntConsolidateStoreRuns = debugCounters.registerCounter(PACKAGE,
				"consolidate-store-runs",
				"Number of times the task to consolidate entries in the " +
				"store witch live known devices has been run");
		cntConsolidateStoreDevicesRemoved = debugCounters.registerCounter(PACKAGE,
				"consolidate-store-devices-removed",
				"Number of times a device has been removed from the sync " +
						"store because no corresponding live device is known. " +
						"This indicates a remote controller still writing device " +
						"entries despite the local controller being MASTER or an " +
						"incosistent store update from the local controller.", IDebugCounterService.MetaData.WARN);
		cntTransitionToMaster = debugCounters.registerCounter(PACKAGE,
				"transition-to-master",
				"Number of times this controller has transitioned from SLAVE " +
				"to MASTER role. Will be 0 or 1.");
	}
	protected class HAListenerDelegate implements IHAListener {
		@Override
		public void transitionToActive() {
			DeviceManagerImpl.this.isMaster = true;
			DeviceManagerImpl.this.deviceSyncManager.goToMaster();
		}
		@Override
		public void controllerNodeIPsChanged(
				Map<String, String> curControllerNodeIPs,
				Map<String, String> addedControllerNodeIPs,
				Map<String, String> removedControllerNodeIPs) {
		}
		@Override
		public String getName() {
			return DeviceManagerImpl.this.getName();
		}
		@Override
		public boolean isCallbackOrderingPrereq(HAListenerTypeMarker type,
				String name) {
			return ("topology".equals(name) ||
					"bvsmanager".equals(name));
		}
		@Override
		public boolean isCallbackOrderingPostreq(HAListenerTypeMarker type,
				String name) {
			return false;
		}
		@Override
		public void transitionToStandby() {
			DeviceManagerImpl.this.isMaster = false;
		}
	}
	protected Command processPacketInMessage(IOFSwitch sw, OFPacketIn pi, FloodlightContext cntx) {
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx,IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
		OFPort inPort = (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT));
		Entity srcEntity = getSourceEntityFromPacket(eth, sw.getId(), inPort);
		if (srcEntity == null) {
			cntInvalidSource.increment();
			return Command.STOP;
		}
		learnDeviceFromArpResponseData(eth, sw.getId(), inPort);
		Device srcDevice = learnDeviceByEntity(srcEntity);
		if (srcDevice == null) {
			cntNoSource.increment();
			return Command.STOP;
		}
		fcStore.put(cntx, CONTEXT_SRC_DEVICE, srcDevice);
		if (eth.getDestinationMACAddress().getLong() == 0) {
			cntInvalidDest.increment();
			return Command.STOP;
		}
		Entity dstEntity = getDestEntityFromPacket(eth);
		Device dstDevice = null;
		if (dstEntity != null) {
			dstDevice = findDestByEntity(srcDevice.getEntityClass(), dstEntity);
			if (dstDevice != null)
				fcStore.put(cntx, CONTEXT_DST_DEVICE, dstDevice);
			else
				cntNoDest.increment();
		} else {
			cntNoDest.increment();
		}
		if (logger.isTraceEnabled()) {
					new Object[] { pi, sw.getId().toString(), inPort, eth,
					srcDevice, dstDevice });
		}
		snoopDHCPClientName(eth, srcDevice);
		return Command.CONTINUE;
	}
	private void snoopDHCPClientName(Ethernet eth, Device srcDevice) {
		if (! (eth.getPayload() instanceof IPv4) )
			return;
		IPv4 ipv4 = (IPv4) eth.getPayload();
		if (! (ipv4.getPayload() instanceof UDP) )
			return;
		UDP udp = (UDP) ipv4.getPayload();
		if (!(udp.getPayload() instanceof DHCP))
			return;
		DHCP dhcp = (DHCP) udp.getPayload();
		byte opcode = dhcp.getOpCode();
		if (opcode == DHCP.OPCODE_REQUEST) {
			DHCPOption dhcpOption = dhcp.getOption(
					DHCPOptionCode.OptionCode_Hostname);
			if (dhcpOption != null) {
				cntDhcpClientNameSnooped.increment();
				srcDevice.dhcpClientName = new String(dhcpOption.getData());
			}
		}
	}
	public boolean isValidAttachmentPoint(DatapathId switchDPID,
			OFPort switchPort) {
		if (topology.isAttachmentPointPort(switchDPID, switchPort) == false)
			return false;
		if (suppressAPs.contains(new SwitchPort(switchDPID, switchPort)))
			return false;
		return true;
	}
	private IPv4Address getSrcIPv4AddrFromARP(Ethernet eth, MacAddress dlAddr) {
		if (eth.getPayload() instanceof ARP) {
			ARP arp = (ARP) eth.getPayload();
			if ((arp.getProtocolType() == ARP.PROTO_TYPE_IP) && (arp.getSenderHardwareAddress().equals(dlAddr))) {
				return arp.getSenderProtocolAddress();
			}
		}
		return IPv4Address.NONE;
	}
	private IPv6Address getSrcIPv6Addr(Ethernet eth) {
		if (eth.getPayload() instanceof IPv6) {
			IPv6 ipv6 = (IPv6) eth.getPayload();
			return ipv6.getSourceAddress();
		}
		return IPv6Address.NONE;
	}
	protected Entity getSourceEntityFromPacket(Ethernet eth, DatapathId swdpid, OFPort port) {
		MacAddress dlAddr = eth.getSourceMACAddress();
		if (dlAddr.isBroadcast() || dlAddr.isMulticast())
			return null;
		if (dlAddr.getLong() == 0)
			return null;
		VlanVid vlan = VlanVid.ofVlan(eth.getVlanID());
		IPv4Address ipv4Src = getSrcIPv4AddrFromARP(eth, dlAddr);
		IPv6Address ipv6Src = ipv4Src.equals(IPv4Address.NONE) ? getSrcIPv6Addr(eth) : IPv6Address.NONE;
		return new Entity(dlAddr,
				vlan,
				ipv4Src,
				ipv6Src,
				swdpid,
				port,
				new Date());
	}
	protected void learnDeviceFromArpResponseData(Ethernet eth,
			DatapathId swdpid,
			OFPort port) {
		if (!(eth.getPayload() instanceof ARP)) return;
		ARP arp = (ARP) eth.getPayload();
		MacAddress dlAddr = eth.getSourceMACAddress();
		MacAddress senderAddr = arp.getSenderHardwareAddress();
		if (senderAddr.isBroadcast() || senderAddr.isMulticast())
			return;
		if (senderAddr.equals(MacAddress.of(0)))
			return;
		VlanVid vlan = VlanVid.ofVlan(eth.getVlanID());
		IPv4Address nwSrc = arp.getSenderProtocolAddress();
		Entity e =  new Entity(senderAddr,
				nwSrc,
				swdpid,
				port,
				new Date());
		learnDeviceByEntity(e);
	}
	protected Entity getDestEntityFromPacket(Ethernet eth) {
		MacAddress dlAddr = eth.getDestinationMACAddress();
		VlanVid vlan = VlanVid.ofVlan(eth.getVlanID());
		IPv4Address ipv4Dst = IPv4Address.NONE;
		IPv6Address ipv6Dst = IPv6Address.NONE;
		if (dlAddr.isBroadcast() || dlAddr.isMulticast())
			return null;
		if (dlAddr.equals(MacAddress.of(0)))
			return null;
		if (eth.getPayload() instanceof IPv4) {
			IPv4 ipv4 = (IPv4) eth.getPayload();
			ipv4Dst = ipv4.getDestinationAddress();
		} else if (eth.getPayload() instanceof IPv6) {
			IPv6 ipv6 = (IPv6) eth.getPayload();
			ipv6Dst = ipv6.getDestinationAddress();
		}
		return new Entity(dlAddr,
				vlan,
				ipv4Dst,
				ipv6Dst,
				DatapathId.NONE,
				OFPort.ZERO,
				Entity.NO_DATE);
	}
	protected Device findDeviceByEntity(Entity entity) {
		Long deviceKey = primaryIndex.findByEntity(entity);
		IEntityClass entityClass = null;
		if (deviceKey == null) {
			entityClass = entityClassifier.classifyEntity(entity);
			if (entityClass == null) {
				return null;
			}
			ClassState classState = getClassState(entityClass);
			if (classState.classIndex != null) {
				deviceKey = classState.classIndex.findByEntity(entity);
			}
		}
		if (deviceKey == null) return null;
		return deviceMap.get(deviceKey);
	}
	protected Device findDestByEntity(IEntityClass reference,
			Entity dstEntity) {
		Long deviceKey = primaryIndex.findByEntity(dstEntity);
		if (deviceKey == null) {
			ClassState classState = getClassState(reference);
			if (classState.classIndex == null) {
				return null;
			}
			deviceKey = classState.classIndex.findByEntity(dstEntity);
		}
		if (deviceKey == null) return null;
		return deviceMap.get(deviceKey);
	}
    private Device findDeviceInClassByEntity(IEntityClass clazz,
                                               Entity entity) {
        throw new UnsupportedOperationException();
    }
	protected Device learnDeviceByEntity(Entity entity) {
		ArrayList<Long> deleteQueue = null;
		LinkedList<DeviceUpdate> deviceUpdates = null;
		Device device = null;
		while (true) {
			deviceUpdates = null;
			Long deviceKey = primaryIndex.findByEntity(entity);
			IEntityClass entityClass = null;
			if (deviceKey == null) {
				entityClass = entityClassifier.classifyEntity(entity);
				if (entityClass == null) {
					device = null;
					break;
				}
				ClassState classState = getClassState(entityClass);
				if (classState.classIndex != null) {
					deviceKey = classState.classIndex.findByEntity(entity);
				}
			}
			if (deviceKey != null) {
				device = deviceMap.get(deviceKey);
				if (device == null) {
					if (logger.isDebugEnabled()) {
						logger.debug("No device for deviceKey {} while "
								+ "while processing entity {}",
								deviceKey, entity);
					}
					continue;
				}
			} else {
				if (entity.hasSwitchPort() && !topology.isAttachmentPointPort(entity.getSwitchDPID(), entity.getSwitchPort())) {
					cntDeviceOnInternalPortNotLearned.increment();
					if (logger.isDebugEnabled()) {
						logger.debug("Not learning new device on internal"
								+ " link: {}", entity);
					}
					device = null;
					break;
				}
				if (!isEntityAllowed(entity, entityClass)) {
					cntPacketNotAllowed.increment();
					if (logger.isDebugEnabled()) {
						logger.debug("PacketIn is not allowed {} {}",
								entityClass.getName(), entity);
					}
					device = null;
					break;
				}
				deviceKey = deviceKeyCounter.getAndIncrement();
				device = allocateDevice(deviceKey, entity, entityClass);
				deviceMap.put(deviceKey, device);
				if (!updateIndices(device, deviceKey)) {
					if (deleteQueue == null)
						deleteQueue = new ArrayList<Long>();
					deleteQueue.add(deviceKey);
					continue;
				}
				updateSecondaryIndices(entity, entityClass, deviceKey);
				cntNewDevice.increment();
				if (logger.isDebugEnabled()) {
					logger.debug("New device created: {} deviceKey={}, entity={}",
							new Object[]{device, deviceKey, entity});
				}
				deviceUpdates = updateUpdates(deviceUpdates, new DeviceUpdate(device, ADD, null));
				break;
			}
			if (!isEntityAllowed(entity, device.getEntityClass())) {
				cntPacketNotAllowed.increment();
				if (logger.isDebugEnabled()) {
					logger.info("PacketIn is not allowed {} {}",
							device.getEntityClass().getName(), entity);
				}
				return null;
			}
			if (entity.hasSwitchPort() && !topology.isAttachmentPointPort(entity.getSwitchDPID(), entity.getSwitchPort())) {
				cntPacketOnInternalPortForKnownDevice.increment();
				break;
			}
			int entityindex = -1;
			if ((entityindex = device.entityIndex(entity)) >= 0) {
				Date lastSeen = entity.getLastSeenTimestamp();
				if (lastSeen.equals(Entity.NO_DATE)) {
					lastSeen = new Date();
					entity.setLastSeenTimestamp(lastSeen);
				}
				device.entities[entityindex].setLastSeenTimestamp(lastSeen);
			} else {
				entityindex = -(entityindex + 1);
				Device newDevice = allocateDevice(device, entity, entityindex);
				EnumSet<DeviceField> changedFields = findChangedFields(device, entity);
				boolean res = deviceMap.replace(deviceKey, device, newDevice);
				if (!res)
					continue;
				device = newDevice;
				if (!updateIndices(device, deviceKey)) {
					continue;
				}
				updateSecondaryIndices(entity,
						device.getEntityClass(),
						deviceKey);
				cntNewEntity.increment();
				if (changedFields.size() > 0) {
					cntDeviceChanged.increment();
					deviceUpdates =
							updateUpdates(deviceUpdates,
									new DeviceUpdate(newDevice, CHANGE,
											changedFields));
				}
			}
			if (entity.hasSwitchPort()) {
				boolean moved = device.updateAttachmentPoint(entity.getSwitchDPID(),
						entity.getSwitchPort(),
						entity.getLastSeenTimestamp());
				if (moved) {
					if (logger.isTraceEnabled()) {
						logger.trace("Device moved: attachment points {}," +
								"entities {}", device.attachmentPoints,
								device.entities);
					}
				} else {
					if (logger.isTraceEnabled()) {
						logger.trace("Device attachment point updated: " +
								"attachment points {}," +
								"entities {}", device.attachmentPoints,
								device.entities);
					}
				}
			}
			break;
		}
		if (deleteQueue != null) {
			for (Long l : deleteQueue) {
				Device dev = deviceMap.get(l);
				this.deleteDevice(dev);
			}
		}
		processUpdates(deviceUpdates);
		deviceSyncManager.storeDeviceThrottled(device);
		return device;
	}
	protected boolean isEntityAllowed(Entity entity, IEntityClass entityClass) {
		return true;
	}
	protected EnumSet<DeviceField> findChangedFields(Device device,
			Entity newEntity) {
		EnumSet<DeviceField> changedFields =
				EnumSet.of(DeviceField.IPv4,
						DeviceField.IPv6,
						DeviceField.VLAN,
						DeviceField.SWITCH);
		if (newEntity.getIpv4Address().equals(IPv4Address.NONE))
			changedFields.remove(DeviceField.IPv4);
		if (newEntity.getIpv6Address().equals(IPv6Address.NONE))
			changedFields.remove(DeviceField.IPv6);
			changedFields.remove(DeviceField.VLAN);
		if (newEntity.getSwitchDPID().equals(DatapathId.NONE) ||
				newEntity.getSwitchPort().equals(OFPort.ZERO))
			changedFields.remove(DeviceField.SWITCH); 
		for (Entity entity : device.getEntities()) {
				changedFields.remove(DeviceField.IPv4);
					entity.getIpv6Address().equals(newEntity.getIpv6Address()))
				changedFields.remove(DeviceField.IPv6);
				changedFields.remove(DeviceField.VLAN);
			if (newEntity.getSwitchDPID().equals(DatapathId.NONE) ||
					newEntity.getSwitchPort().equals(OFPort.ZERO) ||
					(entity.getSwitchDPID().equals(newEntity.getSwitchDPID()) &&
					entity.getSwitchPort().equals(newEntity.getSwitchPort())))
				changedFields.remove(DeviceField.SWITCH);
		}
		if (changedFields.contains(DeviceField.SWITCH)) {
			if (!isValidAttachmentPoint(newEntity.getSwitchDPID(), newEntity.getSwitchPort())) {
				changedFields.remove(DeviceField.SWITCH);
			}
		}
		return changedFields;
	}
	 protected void processUpdates(Queue<DeviceUpdate> updates) {
		if (updates == null) return;
		DeviceUpdate update = null;
		while (null != (update = updates.poll())) {
			if (logger.isTraceEnabled()) {
				logger.trace("Dispatching device update: {}", update);
			}
			if (update.change == DeviceUpdate.Change.DELETE) {
				deviceSyncManager.removeDevice(update.device);
			} else {
				deviceSyncManager.storeDevice(update.device);
			}
			List<IDeviceListener> listeners = deviceListeners.getOrderedListeners();
			notifyListeners(listeners, update);
		}
	 }
	 protected void notifyListeners(List<IDeviceListener> listeners, DeviceUpdate update) {
		 if (listeners == null) {
			 return;
		 }
		 for (IDeviceListener listener : listeners) {
			 switch (update.change) {
			 case ADD:
				 listener.deviceAdded(update.device);
				 break;
			 case DELETE:
				 listener.deviceRemoved(update.device);
				 break;
			 case CHANGE:
				 for (DeviceField field : update.fieldsChanged) {
					 switch (field) {
					 case IPv4:
						 listener.deviceIPV4AddrChanged(update.device);
						 break;
					 case IPv6:
						 listener.deviceIPV6AddrChanged(update.device);
						 break;
					 case SWITCH:
					 case PORT:
						 break;
					 case VLAN:
						 listener.deviceVlanChanged(update.device);
						 break;
					 default:
						 logger.debug("Unknown device field changed {}",
								 update.fieldsChanged.toString());
						 break;
					 }
				 }
				 break;
			 }
		 }
	 }
	 protected boolean allKeyFieldsPresent(Entity e, EnumSet<DeviceField> keyFields) {
		 for (DeviceField f : keyFields) {
			 switch (f) {
			 case MAC:
				 break;
			 case IPv4:
			 case IPv6:
				 if (e.ipv4Address.equals(IPv4Address.NONE) && e.ipv6Address.equals(IPv6Address.NONE)) {
				 }
				 break;
			 case SWITCH:
				 if (e.switchDPID.equals(DatapathId.NONE)) {
					 return false;
				 }
				 break;
			 case PORT:
				 if (e.switchPort.equals(OFPort.ZERO)) {
					 return false;
				 }
				 break;
			 case VLAN:
				 }
				 break;
			 default:
				 throw new IllegalStateException();
			 }
		 }
		 return true;
	 }
	 private LinkedList<DeviceUpdate> updateUpdates(LinkedList<DeviceUpdate> list, DeviceUpdate update) {
		 if (update == null) return list;
		 if (list == null)
			 list = new LinkedList<DeviceUpdate>();
		 list.add(update);
		 return list;
	 }
	 private ClassState getClassState(IEntityClass clazz) {
		 ClassState classState = classStateMap.get(clazz.getName());
		 if (classState != null) return classState;
		 classState = new ClassState(clazz);
		 ClassState r = classStateMap.putIfAbsent(clazz.getName(), classState);
		 if (r != null) {
			 return r;
		 }
		 return classState;
	 }
	 private boolean updateIndices(Device device, Long deviceKey) {
		 if (!primaryIndex.updateIndex(device, deviceKey)) {
			 return false;
		 }
		 IEntityClass entityClass = device.getEntityClass();
		 ClassState classState = getClassState(entityClass);
		 if (classState.classIndex != null) {
			 if (!classState.classIndex.updateIndex(device,
					 deviceKey))
				 return false;
		 }
		 return true;
	 }
	 private void updateSecondaryIndices(Entity entity,
			 IEntityClass entityClass,
			 Long deviceKey) {
		 for (DeviceIndex index : secondaryIndexMap.values()) {
			 index.updateIndex(entity, deviceKey);
		 }
		 ClassState state = getClassState(entityClass);
		 for (DeviceIndex index : state.secondaryIndexMap.values()) {
			 index.updateIndex(entity, deviceKey);
		 }
	 }
	 protected void cleanupEntities () {
		 cntCleanupEntitiesRuns.increment();
		 Calendar c = Calendar.getInstance();
		 c.add(Calendar.MILLISECOND, -ENTITY_TIMEOUT);
		 Date cutoff = c.getTime();
		 ArrayList<Entity> toRemove = new ArrayList<Entity>();
		 ArrayList<Entity> toKeep = new ArrayList<Entity>();
		 Iterator<Device> diter = deviceMap.values().iterator();
		 LinkedList<DeviceUpdate> deviceUpdates =
				 new LinkedList<DeviceUpdate>();
		 while (diter.hasNext()) {
			 Device d = diter.next();
			 while (true) {
				 deviceUpdates.clear();
				 toRemove.clear();
				 toKeep.clear();
				 for (Entity e : d.getEntities()) {
					 if (!e.getLastSeenTimestamp().equals(Entity.NO_DATE) &&
							 0 > e.getLastSeenTimestamp().compareTo(cutoff)) {
						 toRemove.add(e);
					 } else {
						 toKeep.add(e);
					 }
				 }
				 if (toRemove.size() == 0) {
					 break;
				 }
				 cntEntityRemovedTimeout.increment();
				 for (Entity e : toRemove) {
					 removeEntity(e, d.getEntityClass(), d.getDeviceKey(), toKeep);
				 }
				 if (toKeep.size() > 0) {
					 Device newDevice = allocateDevice(d.getDeviceKey(),
							 d.getDHCPClientName(),
							 d.oldAPs,
							 d.attachmentPoints,
							 toKeep,
							 d.getEntityClass());
					 EnumSet<DeviceField> changedFields =
							 EnumSet.noneOf(DeviceField.class);
					 for (Entity e : toRemove) {
						 changedFields.addAll(findChangedFields(newDevice, e));
					 }
					 DeviceUpdate update = null;
					 if (changedFields.size() > 0) {
						 update = new DeviceUpdate(d, CHANGE, changedFields);
					 }
					 if (!deviceMap.replace(newDevice.getDeviceKey(),
							 d,
							 newDevice)) {
						 d = deviceMap.get(d.getDeviceKey());
								 if (null != d)
									 continue;
					 }
					 if (update != null) {
						 cntDeviceChanged.increment();
						 deviceUpdates.add(update);
					 }
				 } else {
					 DeviceUpdate update = new DeviceUpdate(d, DELETE, null);
					 if (!deviceMap.remove(d.getDeviceKey(), d)) {
						 d = deviceMap.get(d.getDeviceKey());
						 if (null != d)
							 continue;
						 cntDeviceDeleted.increment();
					 }
					 deviceUpdates.add(update);
				 }
				 processUpdates(deviceUpdates);
				 break;
			 }
		 }
		 debugEventService.flushEvents();
	 }
	 protected void removeEntity(Entity removed,
			 IEntityClass entityClass,
			 Long deviceKey,
			 Collection<Entity> others) {
		 for (DeviceIndex index : secondaryIndexMap.values()) {
			 index.removeEntityIfNeeded(removed, deviceKey, others);
		 }
		 ClassState classState = getClassState(entityClass);
		 for (DeviceIndex index : classState.secondaryIndexMap.values()) {
			 index.removeEntityIfNeeded(removed, deviceKey, others);
		 }
		 primaryIndex.removeEntityIfNeeded(removed, deviceKey, others);
		 if (classState.classIndex != null) {
			 classState.classIndex.removeEntityIfNeeded(removed,
					 deviceKey,
					 others);
		 }
	 }
	 protected void deleteDevice(Device device) {
		 ArrayList<Entity> emptyToKeep = new ArrayList<Entity>();
		 for (Entity entity : device.getEntities()) {
			 this.removeEntity(entity, device.getEntityClass(),
					 device.getDeviceKey(), emptyToKeep);
		 }
		 if (!deviceMap.remove(device.getDeviceKey(), device)) {
			 if (logger.isDebugEnabled())
				 logger.debug("device map does not have this device -" +
						 device.toString());
		 }
	 }
	 private EnumSet<DeviceField> getEntityKeys(@Nonnull MacAddress macAddress,
			 @Nonnull IPv4Address ipv4Address,
			 @Nonnull IPv6Address ipv6Address,
			 @Nonnull DatapathId switchDPID,
			 @Nonnull OFPort switchPort) {
		 EnumSet<DeviceField> keys = EnumSet.noneOf(DeviceField.class);
		 if (!macAddress.equals(MacAddress.NONE)) keys.add(DeviceField.MAC);
		 if (!ipv4Address.equals(IPv4Address.NONE)) keys.add(DeviceField.IPv4);
		 if (!ipv6Address.equals(IPv6Address.NONE)) keys.add(DeviceField.IPv6);
		 if (!switchDPID.equals(DatapathId.NONE)) keys.add(DeviceField.SWITCH);
		 if (!switchPort.equals(OFPort.ZERO)) keys.add(DeviceField.PORT);
		 return keys;
	 }
	 protected Iterator<Device> queryClassByEntity(IEntityClass clazz,
			 EnumSet<DeviceField> keyFields,
			 Entity entity) {
		 ClassState classState = getClassState(clazz);
		 DeviceIndex index = classState.secondaryIndexMap.get(keyFields);
		 if (index == null) return Collections.<Device>emptySet().iterator();
		 return new DeviceIndexInterator(this, index.queryByEntity(entity));
	 }
	 protected Device allocateDevice(Long deviceKey,
			 Entity entity,
			 IEntityClass entityClass) {
		 return new Device(this, deviceKey, entity, entityClass);
	 }
	 protected Device allocateDevice(Long deviceKey,
			 String dhcpClientName,
			 List<AttachmentPoint> aps,
			 List<AttachmentPoint> trueAPs,
			 Collection<Entity> entities,
			 IEntityClass entityClass) {
		 return new Device(this, deviceKey, dhcpClientName, aps, trueAPs,
				 entities, entityClass);
	 }
	 protected Device allocateDevice(Device device,
			 Entity entity,
			 int insertionpoint) {
		 return new Device(device, entity, insertionpoint);
	 }
	 protected Device allocateDevice(Device device, Set <Entity> entities) {
		 List <AttachmentPoint> newPossibleAPs =
				 new ArrayList<AttachmentPoint>();
		 List <AttachmentPoint> newAPs =
				 new ArrayList<AttachmentPoint>();
		 for (Entity entity : entities) {
			 if (entity.switchDPID != null && entity.switchPort != null) {
				 AttachmentPoint aP =
						 new AttachmentPoint(entity.switchDPID,
								 entity.switchPort, new Date(0));
				 newPossibleAPs.add(aP);
			 }
		 }
		 if (device.attachmentPoints != null) {
			 for (AttachmentPoint oldAP : device.attachmentPoints) {
				 if (newPossibleAPs.contains(oldAP)) {
					 newAPs.add(oldAP);
				 }
			 }
		 }
		 if (newAPs.isEmpty())
			 newAPs = null;
		 Device d = new Device(this, device.getDeviceKey(),
				 device.getDHCPClientName(), newAPs, null,
				 entities, device.getEntityClass());
		 d.updateAttachmentPoint();
		 return d;
	 @Override
	 public void topologyChanged(List<LDUpdate> updateList) {
		 Iterator<Device> diter = deviceMap.values().iterator();
		 if (updateList != null) {
			 if (logger.isTraceEnabled()) {
				 for(LDUpdate update: updateList) {
					 logger.trace("Topo update: {}", update);
				 }
			 }
		 }
		 while (diter.hasNext()) {
			 Device d = diter.next();
			 if (d.updateAttachmentPoint()) {
				 if (logger.isDebugEnabled()) {
					 logger.debug("Attachment point changed for device: {}", d);
				 }
				 sendDeviceMovedNotification(d);
			 }
		 }
		 debugEventService.flushEvents();
	 }
	 protected void sendDeviceMovedNotification(Device d) {
		 cntDeviceMoved.increment();
		 deviceSyncManager.storeDevice(d);
		 List<IDeviceListener> listeners = deviceListeners.getOrderedListeners();
		 if (listeners != null) {
			 for (IDeviceListener listener : listeners) {
				 listener.deviceMoved(d);
			 }
		 }
	 }
	 @Override
	 public void entityClassChanged (Set<String> entityClassNames) {
		 Iterator<Device> diter = deviceMap.values().iterator();
		 while (diter.hasNext()) {
			 Device d = diter.next();
			 if (d.getEntityClass() == null ||
					 entityClassNames.contains(d.getEntityClass().getName()))
				 reclassifyDevice(d);
		 }
	 }
	 protected boolean reclassifyDevice(Device device)
	 {
		 if (device == null) {
			 logger.debug("In reclassify for null device");
			 return false;
		 }
		 boolean needToReclassify = false;
		 for (Entity entity : device.entities) {
			 IEntityClass entityClass =
					 this.entityClassifier.classifyEntity(entity);
			 if (entityClass == null || device.getEntityClass() == null) {
				 needToReclassify = true;
				 break;
			 }
			 if (!entityClass.getName().
					 equals(device.getEntityClass().getName())) {
				 needToReclassify = true;
				 break;
			 }
		 }
		 if (needToReclassify == false) {
			 return false;
		 }
		 cntDeviceReclassifyDelete.increment();
		 LinkedList<DeviceUpdate> deviceUpdates =
				 new LinkedList<DeviceUpdate>();
		 this.deleteDevice(device);
		 deviceUpdates.add(new DeviceUpdate(device,
				 DeviceUpdate.Change.DELETE, null));
		 if (!deviceUpdates.isEmpty())
			 processUpdates(deviceUpdates);
		 for (Entity entity: device.entities ) {
			 this.learnDeviceByEntity(entity);
		 }
		 debugEventService.flushEvents();
		 return true;
	 }
	 void setSyncStoreWriteInterval(int intervalMs) {
		 this.syncStoreWriteIntervalMs = intervalMs;
	 }
	 void setInitialSyncStoreConsolidateMs(int intervalMs) {
		 this.initialSyncStoreConsolidateMs = intervalMs;
	 }
	 void scheduleConsolidateStoreNow() {
		 this.storeConsolidateTask.reschedule(0, TimeUnit.MILLISECONDS);
	 }
	 private class DeviceSyncManager  {
		 private final ConcurrentMap<Long, Long> lastWriteTimes = new ConcurrentHashMap<Long, Long>();
		 public void storeDevice(Device d) {
			 if (!isMaster)
				 return;
			 if (d == null)
				 return;
			 long now = System.nanoTime();
			 writeUpdatedDeviceToStorage(d);
			 lastWriteTimes.put(d.getDeviceKey(), now);
		 }
		 public void storeDeviceThrottled(Device d) {
			 if (!isMaster)
				 return;
			 if (d == null)
				 return;
			 long now = System.nanoTime();
			 Long last = lastWriteTimes.get(d.getDeviceKey());
			 if (last == null || (now - last) > intervalNs) {
				 writeUpdatedDeviceToStorage(d);
				 lastWriteTimes.put(d.getDeviceKey(), now);
			 } else {
				 cntDeviceStoreThrottled.increment();
			 }
		 }
		 public void removeDevice(Device d) {
			 if (!isMaster)
				 return;
			 lastWriteTimes.remove(d.getDeviceKey());
			 try {
				 cntDeviceRemovedFromStore.increment();
				 storeClient.delete(DeviceSyncRepresentation.computeKey(d));
			 } catch(ObsoleteVersionException e) {
			 } catch (SyncException e) {
				 cntSyncException.increment();
				 logger.error("Could not remove device " + d + " from store", e);
			 }
		 }
		 private void removeDevice(Versioned<DeviceSyncRepresentation> dev) {
			 try {
				 cntDeviceRemovedFromStore.increment();
				 storeClient.delete(dev.getValue().getKey(),
						 dev.getVersion());
			 } catch(ObsoleteVersionException e) {
			 } catch(SyncException e) {
				 cntSyncException.increment();
				 logger.error("Failed to remove device entry for " +
						 dev.toString() + " from store.", e);
			 }
		 }
		 private void goToMaster() {
			 if (logger.isDebugEnabled()) {
				 logger.debug("Transitioning to MASTER role");
			 }
			 cntTransitionToMaster.increment();
			 IClosableIterator<Map.Entry<String,Versioned<DeviceSyncRepresentation>>>
			 iter = null;
			 try {
				 iter = storeClient.entries();
			 } catch (SyncException e) {
				 cntSyncException.increment();
				 logger.error("Failed to read devices from sync store", e);
				 return;
			 }
			 try {
				 while(iter.hasNext()) {
					 Versioned<DeviceSyncRepresentation> versionedDevice =
							 iter.next().getValue();
					 DeviceSyncRepresentation storedDevice =
							 versionedDevice.getValue();
					 if (storedDevice == null)
						 continue;
					 cntDevicesFromStore.increment();
					 for(SyncEntity se: storedDevice.getEntities()) {
						 learnDeviceByEntity(se.asEntity());
					 }
				 }
			 } finally {
				 if (iter != null)
					 iter.close();
			 }
			 storeConsolidateTask.reschedule(initialSyncStoreConsolidateMs,
					 TimeUnit.MILLISECONDS);
		 }
		 private void writeUpdatedDeviceToStorage(Device device) {
			 try {
				 cntDeviceStrored.increment();
				 DeviceSyncRepresentation storeDevice = new DeviceSyncRepresentation(device);
				 storeClient.put(storeDevice.getKey(), storeDevice);
			 } catch (ObsoleteVersionException e) {
			 } catch (SyncException e) {
				 cntSyncException.increment();
				 logger.error("Could not write device " + device +
						 " to sync store:", e);
			 } catch (Exception e) {
				 logger.error("Count not write device to sync storage " + e.getMessage());
			 }
		 }
		 private void consolidateStore() {
			 if (!isMaster)
				 return;
			 cntConsolidateStoreRuns.increment();
			 if (logger.isDebugEnabled()) {
				 logger.debug("Running consolidateStore.");
			 }
			 IClosableIterator<Map.Entry<String,Versioned<DeviceSyncRepresentation>>>
			 iter = null;
			 try {
				 iter = storeClient.entries();
			 } catch (SyncException e) {
				 cntSyncException.increment();
				 logger.error("Failed to read devices from sync store", e);
				 return;
			 }
			 try {
				 while(iter.hasNext()) {
					 boolean found = false;
					 Versioned<DeviceSyncRepresentation> versionedDevice =
							 iter.next().getValue();
					 DeviceSyncRepresentation storedDevice =
							 versionedDevice.getValue();
					 if (storedDevice == null)
						 continue;
					 for(SyncEntity se: storedDevice.getEntities()) {
						 try {
									 IDevice d = findDevice(MacAddress.of(se.macAddress), VlanVid.ofVlan(se.vlan),
											 IPv4Address.of(se.ipv4Address),
											 IPv6Address.NONE,
											 DatapathId.of(se.switchDPID),
											 OFPort.of(se.switchPort));
									 if (d != null) {
										 found = true;
										 break;
									 }
						 } catch (IllegalArgumentException e) {
						 }
					 }
					 if (!found) {
						 if (logger.isDebugEnabled()) {
							 logger.debug("Removing device {} from store. No "
									 + "corresponding live device",
									 storedDevice.getKey());
						 }
						 cntConsolidateStoreDevicesRemoved.increment();
						 removeDevice(versionedDevice);
					 }
				 }
			 } finally {
				 if (iter != null)
					 iter.close();
			 }
		 }
	 }
	 protected void setSyncServiceIfNotSet(ISyncService syncService) {
		 if (this.syncService == null)
			 this.syncService = syncService;
	 }
	 IHAListener getHAListener() {
		 return this.haListenerDelegate;
	 }
	 private class DeviceEvent {
		 @EventColumn(name = "MAC", description = EventFieldType.MAC)
		 private final MacAddress macAddress;
		 @EventColumn(name = "IPv4s", description = EventFieldType.IPv4)
		 private final List<IPv4Address> ipv4Addresses;
		 @EventColumn(name = "IPv6s", description = EventFieldType.IPv6)
		 private final List<IPv6Address> ipv6Addresses;
		 @EventColumn(name = "Old Attachment Points",
				 description = EventFieldType.COLLECTION_ATTACHMENT_POINT)
		 private final List<SwitchPort> oldAttachmentPoints;
		 @EventColumn(name = "Current Attachment Points",
				 description = EventFieldType.COLLECTION_ATTACHMENT_POINT)
		 private final List<SwitchPort> currentAttachmentPoints;
		 @EventColumn(name = "VLAN IDs", description = EventFieldType.COLLECTION_OBJECT)
		 private final List<VlanVid> vlanIds;
		 @EventColumn(name = "Reason", description = EventFieldType.STRING)
		 private final String reason;
		 public DeviceEvent(MacAddress macAddress, List<IPv4Address> ipv4Addresses,
				 List<IPv6Address> ipv6Addresses,
				 List<SwitchPort> oldAttachmentPoints,
				 List<SwitchPort> currentAttachmentPoints,
				 List<VlanVid> vlanIds, String reason) {
			 super();
			 this.macAddress = macAddress;
			 this.ipv4Addresses = ipv4Addresses;
			 this.ipv6Addresses = ipv6Addresses;
			 this.oldAttachmentPoints = oldAttachmentPoints;
			 this.currentAttachmentPoints = currentAttachmentPoints;
			 this.vlanIds = vlanIds;
			 this.reason = reason;
		 }
	 }
}