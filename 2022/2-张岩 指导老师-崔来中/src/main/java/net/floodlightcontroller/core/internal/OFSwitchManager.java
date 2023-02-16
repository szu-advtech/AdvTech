package net.floodlightcontroller.core.internal;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CopyOnWriteArraySet;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.HAListenerTypeMarker;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IOFConnectionBackend;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitch.SwitchStatus;
import net.floodlightcontroller.core.IOFSwitchBackend;
import net.floodlightcontroller.core.IOFSwitchDriver;
import net.floodlightcontroller.core.IOFSwitchListener;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.SwitchDescription;
import net.floodlightcontroller.core.SwitchSyncRepresentation;
import net.floodlightcontroller.core.internal.Controller.IUpdate;
import net.floodlightcontroller.core.internal.Controller.ModuleLoaderState;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.rest.SwitchRepresentation;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.debugevent.IDebugEventService.EventType;
import net.floodlightcontroller.debugevent.IEventCategory;
import net.floodlightcontroller.debugevent.MockDebugEventService;
import org.projectfloodlight.openflow.protocol.OFControllerRole;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U32;
import org.sdnplatform.sync.IStoreClient;
import org.sdnplatform.sync.IStoreListener;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.Versioned;
import org.sdnplatform.sync.error.SyncException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.MappingJsonFactory;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelOption;
import io.netty.channel.group.DefaultChannelGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.util.concurrent.GlobalEventExecutor;
public class OFSwitchManager implements IOFSwitchManager, INewOFConnectionListener, IHAListener, IFloodlightModule, IOFSwitchService, IStoreListener<DatapathId> {
	private static final Logger log = LoggerFactory.getLogger(OFSwitchManager.class);
	private volatile OFControllerRole role;
	private SwitchManagerCounters counters;
	private ISyncService syncService;
	private IStoreClient<DatapathId, SwitchSyncRepresentation> storeClient;
	public static final String SWITCH_SYNC_STORE_NAME = OFSwitchManager.class.getCanonicalName() + ".stateStore";
	private static String keyStorePassword;
	private static String keyStore;
	protected static boolean clearTablesOnInitialConnectAsMaster = false;
	protected static boolean clearTablesOnEachTransitionToMaster = false;
	protected static Map<DatapathId, TableId> forwardToControllerFlowsUpToTableByDpid;
	protected static List<U32> ofBitmaps;
	protected static OFFactory defaultFactory;
	private ConcurrentHashMap<DatapathId, OFSwitchHandshakeHandler> switchHandlers;
	private ConcurrentHashMap<DatapathId, IOFSwitchBackend> switches;
	private ConcurrentHashMap<DatapathId, IOFSwitch> syncedSwitches;
	private ISwitchDriverRegistry driverRegistry;
	private Set<LogicalOFMessageCategory> logicalOFMessageCategories = new CopyOnWriteArraySet<LogicalOFMessageCategory>();
	private final List<IAppHandshakePluginFactory> handshakePlugins = new CopyOnWriteArrayList<IAppHandshakePluginFactory>();
	private int numRequiredConnections = -1;
	protected IEventCategory<SwitchEvent> evSwitch;
	protected Set<IOFSwitchListener> switchListeners;
	private IFloodlightProviderService floodlightProvider;
	private IDebugEventService debugEventService;
	private IDebugCounterService debugCounterService;
	private NioEventLoopGroup bossGroup;
	private NioEventLoopGroup workerGroup;
	private DefaultChannelGroup cg;
	@Override
	public void transitionToActive() {
		this.role = HARole.ACTIVE.getOFRole();
	}
	@Override
	public void transitionToStandby() {
		this.role = HARole.STANDBY.getOFRole();
	}
	@Override public SwitchManagerCounters getCounters() {
		return this.counters;
	}
	private void addUpdateToQueue(IUpdate iUpdate) {
		floodlightProvider.addUpdateToQueue(iUpdate);
	}
	@Override
	public synchronized void switchAdded(IOFSwitchBackend sw) {
		DatapathId dpid = sw.getId();
		IOFSwitchBackend oldSw = this.switches.put(dpid, sw);
		evSwitch.newEventWithFlush(new SwitchEvent(dpid, "connected"));
		if (oldSw == sw)  {
			counters.errorActivatedSwitchNotPresent.increment();
			log.error("Switch {} added twice?", sw);
			return;
		} else if (oldSw != null) {
			counters.switchWithSameDpidActivated.increment();
			log.warn("New switch added {} for already-added switch {}", sw, oldSw);
			oldSw.cancelAllPendingRequests();
			addUpdateToQueue(new SwitchUpdate(dpid, SwitchUpdateType.REMOVED));
			oldSw.disconnect();
		}
		if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) >= 0) {
			if (forwardToControllerFlowsUpToTableByDpid.containsKey(sw.getId())) {
				sw.setMaxTableForTableMissFlow(forwardToControllerFlowsUpToTableByDpid.get(sw.getId()));
			} else {
				sw.setMaxTableForTableMissFlow(forwardToControllerFlowsUpToTable);
			}
		}
	}
	@Override
	public synchronized void switchStatusChanged(IOFSwitchBackend sw, SwitchStatus oldStatus, SwitchStatus newStatus) {
		DatapathId dpid = sw.getId();
		IOFSwitchBackend presentSw = this.switches.get(dpid);
		if (presentSw != sw)  {
			counters.errorActivatedSwitchNotPresent
			.increment();
			log.debug("Switch {} status change but not present in sync manager", sw);
			return;
		}
		evSwitch.newEventWithFlush(new SwitchEvent(dpid,
				String.format("%s -> %s",
						oldStatus,
						newStatus)));
		if(newStatus == SwitchStatus.MASTER  && role != OFControllerRole.ROLE_MASTER) {
			counters.invalidSwitchActivatedWhileSlave.increment();
			log.error("Switch {} activated but controller not MASTER", sw);
			sw.disconnect();
		}
		if(!oldStatus.isVisible() && newStatus.isVisible()) {
			addUpdateToQueue(new SwitchUpdate(dpid, SwitchUpdateType.ADDED));
		} else if((oldStatus.isVisible() && !newStatus.isVisible())) {
			addUpdateToQueue(new SwitchUpdate(dpid, SwitchUpdateType.REMOVED));
		}
		if(oldStatus != SwitchStatus.MASTER && newStatus == SwitchStatus.MASTER ) {
			counters.switchActivated.increment();
			addUpdateToQueue(new SwitchUpdate(dpid,
					SwitchUpdateType.ACTIVATED));
		} else if(oldStatus == SwitchStatus.MASTER && newStatus != SwitchStatus.MASTER ) {
			counters.switchDeactivated.increment();
			addUpdateToQueue(new SwitchUpdate(dpid, SwitchUpdateType.DEACTIVATED));
		}
	}
	@Override
	public synchronized void switchDisconnected(IOFSwitchBackend sw) {
		DatapathId dpid = sw.getId();
		IOFSwitchBackend presentSw = this.switches.get(dpid);
		if (presentSw != sw)  {
			counters.errorActivatedSwitchNotPresent.increment();
			log.warn("Switch {} disconnect but not present in sync manager", sw);
			return;
		}
		counters.switchDisconnected.increment();
		this.switches.remove(dpid);
	}
	@Override public void handshakeDisconnected(DatapathId dpid) {
		this.switchHandlers.remove(dpid);
	}
	public Iterable<IOFSwitch> getActiveSwitches() {
		ImmutableList.Builder<IOFSwitch> builder = ImmutableList.builder();
		for(IOFSwitch sw: switches.values()) {
			if(sw.getStatus().isControllable())
				builder.add(sw);
		}
		return builder.build();
	}
	public Map<DatapathId, IOFSwitch> getAllSwitchMap(boolean showInvisible) {
		if(showInvisible) {
			return ImmutableMap.<DatapathId, IOFSwitch>copyOf(switches);
		} else {
			ImmutableMap.Builder<DatapathId, IOFSwitch> builder = ImmutableMap.builder();
			for(IOFSwitch sw: switches.values()) {
				if(sw.getStatus().isVisible())
					builder.put(sw.getId(), sw);
			}
			return builder.build();
		}
	}
	@Override
	public Map<DatapathId, IOFSwitch> getAllSwitchMap() {
		return getAllSwitchMap(true);
	}
	@Override
	public Set<DatapathId> getAllSwitchDpids() {
		return getAllSwitchMap().keySet();
	}
	public Set<DatapathId> getAllSwitchDpids(boolean showInvisible) {
		return getAllSwitchMap(showInvisible).keySet();
	}
	@Override
	public IOFSwitch getSwitch(DatapathId dpid) {
		return this.switches.get(dpid);
	}
	@Override
	public IOFSwitch getActiveSwitch(DatapathId dpid) {
		IOFSwitchBackend sw = this.switches.get(dpid);
		if(sw != null && sw.getStatus().isVisible())
			return sw;
		else
			return null;
	}
	enum SwitchUpdateType {
		ADDED,
		REMOVED,
		ACTIVATED,
		DEACTIVATED,
		PORTCHANGED,
		OTHERCHANGE
	}
	class SwitchUpdate implements IUpdate {
		private final DatapathId swId;
		private final SwitchUpdateType switchUpdateType;
		private final OFPortDesc port;
		private final PortChangeType changeType;
		public SwitchUpdate(DatapathId swId, SwitchUpdateType switchUpdateType) {
			this(swId, switchUpdateType, null, null);
		}
		public SwitchUpdate(DatapathId swId,
				SwitchUpdateType switchUpdateType,
				OFPortDesc port,
				PortChangeType changeType) {
			if (switchUpdateType == SwitchUpdateType.PORTCHANGED) {
				if (port == null) {
					throw new NullPointerException("Port must not be null " +
							"for PORTCHANGED updates");
				}
				if (changeType == null) {
					throw new NullPointerException("ChangeType must not be " +
							"null for PORTCHANGED updates");
				}
			} else {
				if (port != null || changeType != null) {
					throw new IllegalArgumentException("port and changeType " +
							"must be null for " + switchUpdateType +
							" updates");
				}
			}
			this.swId = swId;
			this.switchUpdateType = switchUpdateType;
			this.port = port;
			this.changeType = changeType;
		}
		@Override
		public void dispatch() {
			if (log.isTraceEnabled()) {
				log.trace("Dispatching switch update {} {}", swId, switchUpdateType);
			}
			if (switchListeners != null) {
				for (IOFSwitchListener listener : switchListeners) {
					switch(switchUpdateType) {
					case ADDED:
						listener.switchAdded(swId);
						break;
					case REMOVED:
						listener.switchRemoved(swId);
						break;
					case PORTCHANGED:
						counters.switchPortChanged
						.increment();
						listener.switchPortChanged(swId, port, changeType);
						break;
					case ACTIVATED:
						listener.switchActivated(swId);
						break;
					case DEACTIVATED:
						break;
					case OTHERCHANGE:
						counters.switchOtherChange
						.increment();
						listener.switchChanged(swId);
						break;
					}
				}
			}
		}
	}
	@Override
	public void connectionOpened(IOFConnectionBackend connection, OFFeaturesReply featuresReply) {
		DatapathId dpid = connection.getDatapathId();
		OFAuxId auxId = connection.getAuxId();
		log.debug("{} opened", connection);
		if(auxId.equals(OFAuxId.MAIN)) {
			OFSwitchHandshakeHandler handler =
					new OFSwitchHandshakeHandler(connection, featuresReply, this,
							floodlightProvider.getRoleManager(), floodlightProvider.getTimer());
			OFSwitchHandshakeHandler oldHandler = switchHandlers.put(dpid, handler);
			if(oldHandler != null){
				log.debug("{} is a new main connection, killing old handler connections", connection);
				oldHandler.cleanup();
			}
			handler.beginHandshake();
		} else {
			OFSwitchHandshakeHandler handler = switchHandlers.get(dpid);
			if(handler != null) {
				handler.auxConnectionOpened(connection);
			}
			else {
				log.warn("{} arrived before main connection, closing connection", connection);
				connection.disconnect();
			}
		}
	}
	@Override
	public void addSwitchEvent(DatapathId dpid, String reason, boolean flushNow) {
		if (flushNow)
			evSwitch.newEventWithFlush(new SwitchEvent(dpid, reason));
		else
			evSwitch.newEventNoFlush(new SwitchEvent(dpid, reason));
	}
	@Override
	public synchronized void notifyPortChanged(IOFSwitchBackend sw,
			OFPortDesc port,
			PortChangeType changeType) {
		Preconditions.checkNotNull(sw, "switch must not be null");
		Preconditions.checkNotNull(port, "port must not be null");
		Preconditions.checkNotNull(changeType, "changeType must not be null");
		if (role != OFControllerRole.ROLE_MASTER) {
			counters.invalidPortsChanged.increment();
			return;
		}
		if (!this.switches.containsKey(sw.getId())) {
			counters.invalidPortsChanged.increment();
			return;
		}
		if(sw.getStatus().isVisible()) {
			SwitchUpdate update = new SwitchUpdate(sw.getId(),
					SwitchUpdateType.PORTCHANGED,
					port, changeType);
			addUpdateToQueue(update);
		}
	}
	@Override
	public IOFSwitchBackend getOFSwitchInstance(IOFConnectionBackend connection,
			SwitchDescription description,
			OFFactory factory, DatapathId datapathId) {
		return this.driverRegistry.getOFSwitchInstance(connection, description, factory, datapathId);
	}
	@Override
	public void handleMessage(IOFSwitchBackend sw, OFMessage m, FloodlightContext bContext) {
		floodlightProvider.handleMessage(sw, m, bContext);
	}
	@Override
	public void handleOutgoingMessage(IOFSwitch sw, OFMessage m) {
		floodlightProvider.handleOutgoingMessage(sw, m);
	}
	@Override
	public void addOFSwitchDriver(String manufacturerDescriptionPrefix,
			IOFSwitchDriver driver) {
		this.driverRegistry.addSwitchDriver(manufacturerDescriptionPrefix, driver);
	}
	@Override
	public ImmutableList<OFSwitchHandshakeHandler> getSwitchHandshakeHandlers() {
		return ImmutableList.copyOf(this.switchHandlers.values());
	}
	@Override
	public int getNumRequiredConnections() {
		Preconditions.checkState(numRequiredConnections >= 0, "numRequiredConnections not calculated");
		return numRequiredConnections;
	}
	public Set<LogicalOFMessageCategory> getLogicalOFMessageCategories() {
		return logicalOFMessageCategories;
	}
	private int calcNumRequiredConnections() {
		if(!this.logicalOFMessageCategories.isEmpty()){
			TreeSet<OFAuxId> auxConnections = new TreeSet<OFAuxId>();
			for(LogicalOFMessageCategory category : this.logicalOFMessageCategories){
				auxConnections.add(category.getAuxId());
			}
			OFAuxId first = auxConnections.first();
			OFAuxId last = auxConnections.last();
			if(first.equals(OFAuxId.MAIN)) {
				if(last.getValue() != auxConnections.size() - 1){
					throw new IllegalStateException("Logical OF message categories must maintain contiguous OF Aux Ids! i.e. (0,1,2,3,4,5)");
				}
				return auxConnections.size() - 1;
			} else if(first.equals(OFAuxId.of(1))) {
				if(last.getValue() != auxConnections.size()){
					throw new IllegalStateException("Logical OF message categories must maintain contiguous OF Aux Ids! i.e. (1,2,3,4,5)");
				}
				return auxConnections.size();
			} else {
				throw new IllegalStateException("Logical OF message categories must start at 0 (MAIN) or 1");
			}
		} else {
			return 0;
		}
	}
	@Override
	public void addOFSwitchListener(IOFSwitchListener listener) {
		this.switchListeners.add(listener);
	}
	@Override
	public void removeOFSwitchListener(IOFSwitchListener listener) {
		this.switchListeners.remove(listener);
	}
	@Override
	public void registerLogicalOFMessageCategory(LogicalOFMessageCategory category) {
		logicalOFMessageCategories.add(category);
	}
	@Override
	public boolean isCategoryRegistered(LogicalOFMessageCategory category) {
		return logicalOFMessageCategories.contains(category);
	}
	@Override
	public SwitchRepresentation getSwitchRepresentation(DatapathId dpid) {
		IOFSwitch sw = this.switches.get(dpid);
		OFSwitchHandshakeHandler handler = this.switchHandlers.get(dpid);
		if(sw != null && handler != null) {
			return new SwitchRepresentation(sw, handler);
		}
		return null;
	}
	@Override
	public List<SwitchRepresentation> getSwitchRepresentations() {
		List<SwitchRepresentation> representations = new ArrayList<SwitchRepresentation>();
		for(DatapathId dpid : this.switches.keySet()) {
			SwitchRepresentation representation = getSwitchRepresentation(dpid);
			if(representation != null) {
				representations.add(representation);
			}
		}
		return representations;
	}
	@Override
	public void registerHandshakePlugin(IAppHandshakePluginFactory factory) {
		Preconditions.checkState(floodlightProvider.getModuleLoaderState() == ModuleLoaderState.INIT,
				"handshakeplugins can only be registered when the module loader is in state INIT!");
		handshakePlugins.add(factory);
	}
	@Override
	public List<IAppHandshakePluginFactory> getHandshakePlugins() {
		return handshakePlugins;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>>
	getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IOFSwitchService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService>
	getServiceImpls() {
		Map<Class<? extends IFloodlightService>, IFloodlightService> m =
				new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
		m.put(IOFSwitchService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>>
	getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l = new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IDebugEventService.class);
		l.add(IDebugCounterService.class);
		l.add(ISyncService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProvider = context.getServiceImpl(IFloodlightProviderService.class);
		debugEventService = context.getServiceImpl(IDebugEventService.class);
		debugCounterService = context.getServiceImpl(IDebugCounterService.class);
		syncService = context.getServiceImpl(ISyncService.class);
		switchHandlers = new ConcurrentHashMap<DatapathId, OFSwitchHandshakeHandler>();
		switches = new ConcurrentHashMap<DatapathId, IOFSwitchBackend>();
		syncedSwitches = new ConcurrentHashMap<DatapathId, IOFSwitch>();
		floodlightProvider.getTimer();
		counters = new SwitchManagerCounters(debugCounterService);
		driverRegistry = new NaiveSwitchDriverRegistry(this);
		this.switchListeners = new CopyOnWriteArraySet<IOFSwitchListener>();
		try {
			this.storeClient = this.syncService.getStoreClient(
					SWITCH_SYNC_STORE_NAME,
					DatapathId.class,
					SwitchSyncRepresentation.class);
			this.storeClient.addStoreListener(this);
		} catch (UnknownStoreException e) {
			throw new FloodlightModuleException("Error while setting up sync store client", e);
		Map<String, String> configParams = context.getConfigParams(this);
		String path = configParams.get("keyStorePath");
		String pass = configParams.get("keyStorePassword");
		String useSsl = configParams.get("useSsl");
		if (useSsl == null || path == null || path.isEmpty() || 
				(!useSsl.equalsIgnoreCase("yes") && !useSsl.equalsIgnoreCase("true") &&
						!useSsl.equalsIgnoreCase("yep") && !useSsl.equalsIgnoreCase("ja") &&
						!useSsl.equalsIgnoreCase("stimmt")
						)
				) {
			log.warn("SSL disabled. Using unsecure connections between Floodlight and switches.");
			OFSwitchManager.keyStore = null;
			OFSwitchManager.keyStorePassword = null;
		} else {
			log.info("SSL enabled. Using secure connections between Floodlight and switches.");
			log.info("SSL keystore path: {}, password: {}", path, (pass == null ? "" : pass)); 
			OFSwitchManager.keyStore = path;
			OFSwitchManager.keyStorePassword = (pass == null ? "" : pass);
		}
		String clearInitial = configParams.get("clearTablesOnInitialHandshakeAsMaster");
		String clearLater = configParams.get("clearTablesOnEachTransitionToMaster");
		if (clearInitial == null || clearInitial.isEmpty() || 
				(!clearInitial.equalsIgnoreCase("yes") && !clearInitial.equalsIgnoreCase("true") &&
						!clearInitial.equalsIgnoreCase("yep") && !clearInitial.equalsIgnoreCase("ja") &&
						!clearInitial.equalsIgnoreCase("stimmt"))) {
			log.info("Clear switch flow tables on initial handshake as master: FALSE");
			OFSwitchManager.clearTablesOnInitialConnectAsMaster = false;
		} else {
			log.info("Clear switch flow tables on initial handshake as master: TRUE");
			OFSwitchManager.clearTablesOnInitialConnectAsMaster = true;
		}
		if (clearLater == null || clearLater.isEmpty() || 
				(!clearLater.equalsIgnoreCase("yes") && !clearLater.equalsIgnoreCase("true") &&
						!clearLater.equalsIgnoreCase("yep") && !clearLater.equalsIgnoreCase("ja") &&
						!clearLater.equalsIgnoreCase("stimmt"))) {
			log.info("Clear switch flow tables on each transition to master: FALSE");
			OFSwitchManager.clearTablesOnEachTransitionToMaster = false;
		} else {
			log.info("Clear switch flow tables on each transition to master: TRUE");
			OFSwitchManager.clearTablesOnEachTransitionToMaster = true;
		}
		String defaultFlowsUpToTable = configParams.get("defaultMaxTablesToReceiveTableMissFlow");
		if (defaultFlowsUpToTable == null || defaultFlowsUpToTable.isEmpty()) {
			defaultFlowsUpToTable = configParams.get("defaultMaxTableToReceiveTableMissFlow");
		}
		if (defaultFlowsUpToTable != null && !defaultFlowsUpToTable.isEmpty()) {
			defaultFlowsUpToTable = defaultFlowsUpToTable.toLowerCase().trim();
			try {
				forwardToControllerFlowsUpToTable = TableId.of(defaultFlowsUpToTable.startsWith("0x") 
						? Integer.parseInt(defaultFlowsUpToTable.replaceFirst("0x", ""), 16) 
								: Integer.parseInt(defaultFlowsUpToTable));
				log.info("Setting {} as the default max tables to receive table-miss flow", forwardToControllerFlowsUpToTable.toString());
			} catch (IllegalArgumentException e) {
				log.error("Invalid table ID {} for default max tables to receive table-miss flow. Using pre-set of {}", 
						defaultFlowsUpToTable, forwardToControllerFlowsUpToTable.toString());
			}
		} else {
			log.info("Default max tables to receive table-miss flow not configured. Using {}", forwardToControllerFlowsUpToTable.toString());
		}
		String maxPerDpid = configParams.get("maxTablesToReceiveTableMissFlowPerDpid");
		if (maxPerDpid == null || maxPerDpid.isEmpty()) {
			maxPerDpid = configParams.get("maxTableToReceiveTableMissFlowPerDpid");
		}
		forwardToControllerFlowsUpToTableByDpid = jsonToSwitchTableIdMap(maxPerDpid);
		String protocols = configParams.get("supportedOpenFlowVersions");
		Set<OFVersion> ofVersions = new HashSet<OFVersion>();
		if (protocols != null && !protocols.isEmpty()) {
			protocols = protocols.toLowerCase();
			if (protocols.contains("1.0") || protocols.contains("10")) {
				ofVersions.add(OFVersion.OF_10);
			}
			if (protocols.contains("1.1") || protocols.contains("11")) {
				ofVersions.add(OFVersion.OF_11);
			}
			if (protocols.contains("1.2") || protocols.contains("12")) {
				ofVersions.add(OFVersion.OF_12);
			}
			if (protocols.contains("1.3") || protocols.contains("13")) {
				ofVersions.add(OFVersion.OF_13);
			}
			if (protocols.contains("1.4") || protocols.contains("14")) {
				ofVersions.add(OFVersion.OF_14);
			}
		} else {
			log.warn("Supported OpenFlow versions not specified. Using Loxi-defined {}", OFVersion.values());
			ofVersions.addAll(Arrays.asList(OFVersion.values()));
		}
		if (ofVersions.isEmpty()) {
			throw new IllegalStateException("OpenFlow version list should never be empty at this point. Make sure it's being populated in OFSwitchManager's init function.");
		}
		defaultFactory = computeInitialFactory(ofVersions);
		ofBitmaps = computeOurVersionBitmaps(ofVersions);
	}
	private OFFactory computeInitialFactory(Set<OFVersion> ofVersions) {
		if (ofVersions == null || ofVersions.isEmpty()) {
			throw new IllegalStateException("OpenFlow version list should never be null or empty at this point. Make sure it's set in the OFSwitchManager.");
		}
		OFVersion highest = null;
		for (OFVersion v : ofVersions) {
			if (highest == null) {
				highest = v;
			} else if (v.compareTo(highest) > 0) {
				highest = v;
			}
		}
		return OFFactories.getFactory(highest);
	}
	private List<U32> computeOurVersionBitmaps(Set<OFVersion> ofVersions) {
		if (ofVersions == null || ofVersions.isEmpty()) {
			throw new IllegalStateException("OpenFlow version list should never be null or empty at this point. Make sure it's set in the OFSwitchManager.");
		}
		List<U32> bitmaps = new ArrayList<U32>();
		ArrayList<OFVersion> sortedVersions = new ArrayList<OFVersion>(ofVersions);
		Collections.sort(sortedVersions);
		for (OFVersion v : sortedVersions) {
				bitmaps.add(U32.ofRaw(tempBitmap));
				tempBitmap = 0;
				pos++;
			}
			tempBitmap = tempBitmap | (1 << (v.getWireVersion() % size));
		}
		if (tempBitmap != 0) {
			bitmaps.add(U32.ofRaw(tempBitmap));
		}
		log.info("Computed OpenFlow version bitmap as {}", Arrays.asList(tempBitmap));
		return bitmaps;
	}
	private static Map<DatapathId, TableId> jsonToSwitchTableIdMap(String json) {
		MappingJsonFactory f = new MappingJsonFactory();
		JsonParser jp;
		Map<DatapathId, TableId> retValue = new HashMap<DatapathId, TableId>();
		if (json == null || json.isEmpty()) {
			return retValue;
		}
		try {
			try {
				jp = f.createParser(json);
			} catch (JsonParseException e) {
				throw new IOException(e);
			}
			jp.nextToken();
			if (jp.getCurrentToken() != JsonToken.START_OBJECT) {
				throw new IOException("Expected START_OBJECT");
			}
			while (jp.nextToken() != JsonToken.END_OBJECT) {
				if (jp.getCurrentToken() != JsonToken.FIELD_NAME) {
					throw new IOException("Expected FIELD_NAME");
				}
				String n = jp.getCurrentName();
				jp.nextToken();
				if (jp.getText().equals("")) {
					continue;
				}
				DatapathId dpid;
				try {
					n = n.trim();
					dpid = DatapathId.of(n);
					TableId tablesToGetDefaultFlow;
					String value = jp.getText();
					if (value != null && !value.isEmpty()) {
						value = value.trim().toLowerCase();
						try {
							tablesToGetDefaultFlow = TableId.of(
									value.startsWith("0x") 
									? Integer.parseInt(value.replaceFirst("0x", ""), 16) 
											: Integer.parseInt(value)
							retValue.put(dpid, tablesToGetDefaultFlow);
							log.info("Setting max tables to receive table-miss flow to {} for DPID {}", 
									tablesToGetDefaultFlow.toString(), dpid.toString());
							log.error("Invalid value of {} for max tables to receive table-miss flow for DPID {}. Using default of {}.", value, dpid.toString());
						}
					}
				} catch (NumberFormatException e) {
					log.error("Invalid DPID format {} for max tables to receive table-miss flow for specific DPID. Using default for the intended DPID.", n);
				}
			}
		} catch (IOException e) {
			log.error("Using default for remaining DPIDs. JSON formatting error in max tables to receive table-miss flow for DPID input String: {}", e);
		}
		return retValue;
	}
	@Override
	public void startUp(FloodlightModuleContext context) throws FloodlightModuleException {
		startUpBase(context);
		bootstrapNetty();
	}
	public void startUpBase(FloodlightModuleContext context) throws FloodlightModuleException {
		role = floodlightProvider.getRole().getOFRole();
		floodlightProvider.addHAListener(this);
		loadLogicalCategories();
		registerDebugEvents();
	}
	public void bootstrapNetty() {
		try {
			bossGroup = new NioEventLoopGroup();
			workerGroup = new NioEventLoopGroup();
			ServerBootstrap bootstrap = new ServerBootstrap()
			.group(bossGroup, workerGroup)
			.channel(NioServerSocketChannel.class)
			.option(ChannelOption.SO_REUSEADDR, true)
			.option(ChannelOption.SO_KEEPALIVE, true)
			.option(ChannelOption.TCP_NODELAY, true)
			.option(ChannelOption.SO_SNDBUF, Controller.SEND_BUFFER_SIZE);
			OFChannelInitializer initializer = new OFChannelInitializer(
					this, 
					this, 
					debugCounterService, 
					floodlightProvider.getTimer(), 
					ofBitmaps, 
					defaultFactory, 
					keyStore, 
					keyStorePassword);
			bootstrap.childHandler(initializer);
			cg = new DefaultChannelGroup(GlobalEventExecutor.INSTANCE);
			Set<InetSocketAddress> addrs = new HashSet<InetSocketAddress>();
			if (floodlightProvider.getOFAddresses().isEmpty()) {
				cg.add(bootstrap.bind(new InetSocketAddress(InetAddress.getByAddress(IPv4Address.NONE.getBytes()), floodlightProvider.getOFPort().getPort())).channel());
			} else {
				for (IPv4Address ip : floodlightProvider.getOFAddresses()) {
					addrs.add(new InetSocketAddress(InetAddress.getByAddress(ip.getBytes()), floodlightProvider.getOFPort().getPort()));
				}
			}
			for (InetSocketAddress sa : addrs) {
				cg.add(bootstrap.bind(sa).channel());
				log.info("Listening for switch connections on {}", sa);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	public void loadLogicalCategories() {
		logicalOFMessageCategories = ImmutableSet.copyOf(logicalOFMessageCategories);
		numRequiredConnections = calcNumRequiredConnections();
	}
	private void registerDebugEvents() throws FloodlightModuleException {
		if (debugEventService == null) {
			debugEventService = new MockDebugEventService();
		}
		evSwitch = debugEventService.buildEvent(SwitchEvent.class)
				.setModuleName(this.counters.getPrefix())
				.setEventName("switch-event")
				.setEventDescription("Switch connected, disconnected or port changed")
				.setEventType(EventType.ALWAYS_LOG)
				.setBufferCapacity(100)
				.register();
	}
	@Override
	public String getName() {
		return null;
	}
	@Override
	public boolean isCallbackOrderingPrereq(HAListenerTypeMarker type, String name) {
		return false;
	}
	@Override
	public boolean isCallbackOrderingPostreq(HAListenerTypeMarker type, String name) {
		return false;
	}
	@Override
	public void controllerNodeIPsChanged(Map<String, String> curControllerNodeIPs,
			Map<String, String> addedControllerNodeIPs,
			Map<String, String> removedControllerNodeIPs) {		
	}
	@Override
	public void keysModified(Iterator<DatapathId> keys, UpdateType type) {
		if (type == UpdateType.LOCAL) {
			return;
		}
		while(keys.hasNext()) {
			DatapathId key = keys.next();
			Versioned<SwitchSyncRepresentation> versionedSwitch = null;
			try {
				versionedSwitch = storeClient.get(key);
			} catch (SyncException e) {
				log.error("Exception while retrieving switch " + key.toString() +
						" from sync store. Skipping", e);
				continue;
			}
			if (log.isTraceEnabled()) {
				log.trace("Reveiced switch store notification: key={}, " +
						"entry={}", key, versionedSwitch.getValue());
			}
			if (versionedSwitch.getValue() == null) {
				switchRemovedFromStore(key);
				continue;
			}
			SwitchSyncRepresentation storedSwitch = versionedSwitch.getValue();
			IOFSwitch sw = getSwitch(storedSwitch.getDpid());
			if (!key.equals(storedSwitch.getFeaturesReply(sw.getOFFactory()).getDatapathId())) {
				log.error("Inconsistent DPIDs from switch sync store: " +
						"key is {} but sw.getId() says {}. Ignoring",
						key.toString(), sw.getId());
				continue;
			}
			switchAddedToStore(sw);
		}
	}
	private synchronized void switchRemovedFromStore(DatapathId dpid) {
		if (floodlightProvider.getRole() != HARole.STANDBY) {
		}
		IOFSwitch oldSw = syncedSwitches.remove(dpid);
		if (oldSw != null) {
			addUpdateToQueue(new SwitchUpdate(dpid, SwitchUpdateType.REMOVED));
		} else {
		}
	}
	private synchronized void switchAddedToStore(IOFSwitch sw) {
		if (floodlightProvider.getRole() != HARole.STANDBY) {
		}
		DatapathId dpid = sw.getId();
		IOFSwitch oldSw = syncedSwitches.put(dpid, sw);
		if (oldSw == null)  {
			addUpdateToQueue(new SwitchUpdate(dpid, SwitchUpdateType.ADDED));
		} else {
			sendNotificationsIfSwitchDiffers(oldSw, sw);
		}
	}
	private synchronized void sendNotificationsIfSwitchDiffers(IOFSwitch oldSw, IOFSwitch newSw) {
        for (PortChangeEvent ev: portDiffs) {
            SwitchUpdate update = new SwitchUpdate(newSw.getId(),
                                     SwitchUpdateType.PORTCHANGED,
                                     ev.port, ev.type);
            addUpdateToQueue(update);
	}
}
