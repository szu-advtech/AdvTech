package net.floodlightcontroller.core.internal;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingQueue;
import io.netty.util.HashedWheelTimer;
import io.netty.util.Timer;
import net.floodlightcontroller.core.ControllerId;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.HAListenerTypeMarker;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IControllerCompletionListener;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IInfoProvider;
import net.floodlightcontroller.core.IShutdownService;
import net.floodlightcontroller.core.IListener.Command;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitchListener;
import net.floodlightcontroller.core.LogicalOFMessageCategory;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.RoleInfo;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.FloodlightModuleLoader;
import net.floodlightcontroller.core.util.ListenerDispatcher;
import net.floodlightcontroller.core.web.CoreWebRoutable;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.notification.INotificationManager;
import net.floodlightcontroller.notification.NotificationManagerFactory;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.TransportPort;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.perfmon.IPktInProcessingTimeService;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.storage.IResultSet;
import net.floodlightcontroller.storage.IStorageSourceListener;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.storage.StorageException;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import org.sdnplatform.sync.ISyncService;
import org.sdnplatform.sync.ISyncService.Scope;
import org.sdnplatform.sync.error.SyncException;
import org.sdnplatform.sync.internal.config.ClusterConfig;
import net.floodlightcontroller.util.LoadMonitor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
public class Controller implements IFloodlightProviderService, IStorageSourceListener, IInfoProvider {
    protected static final Logger log = LoggerFactory.getLogger(Controller.class);
    protected static final INotificationManager notifier = NotificationManagerFactory.getNotificationManager(Controller.class);
    static final String ERROR_DATABASE = "The controller could not communicate with the system database.";
    protected ConcurrentMap<OFType, ListenerDispatcher<OFType,IOFMessageListener>> messageListeners;
    protected ConcurrentLinkedQueue<IControllerCompletionListener> completionListeners;
    protected HashMap<String, String> controllerNodeIPsCache;
    protected ListenerDispatcher<HAListenerTypeMarker,IHAListener> haListeners;
    protected Map<String, List<IInfoProvider>> providerMap;
    protected BlockingQueue<IUpdate> updates;
    protected ControllerCounters counters;
    protected Timer timer;
    private ModuleLoaderState moduleLoaderState;
    public enum ModuleLoaderState {
        INIT, STARTUP, COMPLETE
    }
    private IStorageSourceService storageSourceService;
    private IOFSwitchService switchService;
    private IDebugCounterService debugCounterService;
    protected IDebugEventService debugEventService;
    private IRestApiService restApiService;
    private IPktInProcessingTimeService pktinProcTimeService;
    private IThreadPoolService threadPoolService;
    private ISyncService syncService;
    private IShutdownService shutdownService;
	private static Set<IPv4Address> openFlowAddresses = new HashSet<IPv4Address>();
    protected int workerThreads = 16;
    protected String controllerId = "my-floodlight-controller";
    protected volatile HARole notifiedRole;
    private static final String INITIAL_ROLE_CHANGE_DESCRIPTION = "Controller startup.";
    private RoleManager roleManager;
    protected static final String CONTROLLER_TABLE_NAME = "controller_controller";
    protected static final String CONTROLLER_ID = "id";
    protected static final String SWITCH_CONFIG_TABLE_NAME = "controller_switchconfig";
    protected static final String SWITCH_CONFIG_CORE_SWITCH = "core_switch";
    protected static final String CONTROLLER_INTERFACE_TABLE_NAME = "controller_controllerinterface";
    protected static final String CONTROLLER_INTERFACE_ID = "id";
    protected static final String CONTROLLER_INTERFACE_CONTROLLER_ID = "controller_id";
    protected static final String CONTROLLER_INTERFACE_TYPE = "type";
    protected static final String CONTROLLER_INTERFACE_NUMBER = "number";
    protected static final String CONTROLLER_INTERFACE_DISCOVERED_IP = "discovered_ip";
    private static final String FLOW_PRIORITY_TABLE_NAME = "controller_forwardingconfig";
    private static final String FLOW_COLUMN_PRIMARY_KEY = "id";
    private static final String FLOW_VALUE_PRIMARY_KEY = "forwarding";
    private static final String FLOW_COLUMN_ACCESS_PRIORITY = "access_priority";
    private static final String FLOW_COLUMN_CORE_PRIORITY = "core_priority";
    private static final String[] FLOW_COLUMN_NAMES = new String[] {
            FLOW_COLUMN_PRIMARY_KEY,
            FLOW_COLUMN_ACCESS_PRIORITY,
            FLOW_COLUMN_CORE_PRIORITY
    };
    protected static final boolean ALWAYS_DECODE_ETH = true;
    Set<String> uplinkPortPrefixSet;
    @Override
    public Set<String> getUplinkPortPrefixSet() {
        return uplinkPortPrefixSet;
    }
    public void setUplinkPortPrefixSet(Set<String> prefixSet) {
        this.uplinkPortPrefixSet = prefixSet;
    }
    @Override
    public ModuleLoaderState getModuleLoaderState(){
        return this.moduleLoaderState;
    }
    protected final boolean overload_drop = Boolean.parseBoolean(System.getProperty("overload_drop", "false"));
    protected final LoadMonitor loadmonitor = new LoadMonitor(log);
    private static class NotificationSwitchListener implements IOFSwitchListener {
        @Override
        public void switchAdded(DatapathId switchId) {
            notifier.postNotification("Switch " +  switchId + " connected.");
        }
        @Override
        public void switchRemoved(DatapathId switchId) {
            notifier.postNotification("Switch " + switchId + " disconnected.");
        }
        @Override
        public void switchActivated(DatapathId switchId) {
        }
        @Override
        public void switchPortChanged(DatapathId switchId, OFPortDesc port,
                                      PortChangeType type) {
            String msg = String.format("Switch %s port %s changed: %s",
                                       switchId,
                                       port.getName(),
                                       type.toString());
            notifier.postNotification(msg);
        }
        @Override
        public void switchChanged(DatapathId switchId) {
        }
    }
    public interface IUpdate {
        public void dispatch();
    }
    private class HAControllerNodeIPUpdate implements IUpdate {
        public final Map<String,String> curControllerNodeIPs;
        public final Map<String,String> addedControllerNodeIPs;
        public final Map<String,String> removedControllerNodeIPs;
        public HAControllerNodeIPUpdate(
                HashMap<String,String> curControllerNodeIPs,
                HashMap<String,String> addedControllerNodeIPs,
                HashMap<String,String> removedControllerNodeIPs) {
            this.curControllerNodeIPs = curControllerNodeIPs;
            this.addedControllerNodeIPs = addedControllerNodeIPs;
            this.removedControllerNodeIPs = removedControllerNodeIPs;
        }
        @Override
        public void dispatch() {
            if (log.isTraceEnabled()) {
                log.trace("Dispatching HA Controller Node IP update "
                        + "curIPs = {}, addedIPs = {}, removedIPs = {}",
                        new Object[] { curControllerNodeIPs, addedControllerNodeIPs,
                            removedControllerNodeIPs }
                        );
            }
            if (haListeners != null) {
                for (IHAListener listener: haListeners.getOrderedListeners()) {
                    listener.controllerNodeIPsChanged(curControllerNodeIPs,
                            addedControllerNodeIPs, removedControllerNodeIPs);
                }
            }
        }
    }
    void setStorageSourceService(IStorageSourceService storageSource) {
        this.storageSourceService = storageSource;
    }
    IStorageSourceService getStorageSourceService() {
        return this.storageSourceService;
    }
    IShutdownService getShutdownService() {
    	return this.shutdownService;
    }
    void setShutdownService(IShutdownService shutdownService) {
    	this.shutdownService = shutdownService;
    }
    public void setDebugEvent(IDebugEventService debugEvent) {
        this.debugEventService = debugEvent;
    }
    void setDebugCounter(IDebugCounterService debugCounters) {
        this.debugCounterService = debugCounters;
    }
    IDebugCounterService getDebugCounter() {
        return this.debugCounterService;
    }
    void setSyncService(ISyncService syncService) {
        this.syncService = syncService;
    }
    void setPktInProcessingService(IPktInProcessingTimeService pits) {
        this.pktinProcTimeService = pits;
    }
    void setRestApiService(IRestApiService restApi) {
        this.restApiService = restApi;
    }
    void setThreadPoolService(IThreadPoolService tp) {
        this.threadPoolService = tp;
    }
    IThreadPoolService getThreadPoolService() {
        return this.threadPoolService;
    }
    public void setSwitchService(IOFSwitchService switchService) {
       this.switchService = switchService;
    }
    public IOFSwitchService getSwitchService() {
        return this.switchService;
    }
    @Override
    public int getWorkerThreads() {
        return this.workerThreads;
    }
    @Override
    public HARole getRole() {
        return notifiedRole;
    }
    @Override
    public RoleInfo getRoleInfo() {
        return roleManager.getRoleInfo();
    }
    @Override
    public void setRole(HARole role, String changeDescription) {
        roleManager.setRole(role, changeDescription);
    }
    protected static final ThreadLocal<Stack<FloodlightContext>> flcontext_cache =
        new ThreadLocal <Stack<FloodlightContext>> () {
            @Override
            protected Stack<FloodlightContext> initialValue() {
                return new Stack<FloodlightContext>();
            }
        };
    protected static FloodlightContext flcontext_alloc() {
        FloodlightContext flcontext = null;
        if (flcontext_cache.get().empty()) {
            flcontext = new FloodlightContext();
        }
        else {
            flcontext = flcontext_cache.get().pop();
        }
        return flcontext;
    }
    protected void flcontext_free(FloodlightContext flcontext) {
        flcontext.getStorage().clear();
        flcontext_cache.get().push(flcontext);
    }
    @Override
    public void handleMessage(IOFSwitch sw, OFMessage m,
                                 FloodlightContext bContext) {
        Ethernet eth = null;
        log.trace("Dispatching OFMessage to listeners.");
        if (this.notifiedRole == HARole.STANDBY) {
            counters.dispatchMessageWhileStandby.increment();
            return;
        }
        counters.dispatchMessage.increment();
        switch (m.getType()) {
            case PACKET_IN:
            	counters.packetIn.increment();
                OFPacketIn pi = (OFPacketIn)m;
                if (pi.getData().length <= 0) {
                    log.error("Ignoring PacketIn (Xid = " + pi.getXid() + ") because the data field is empty.");
                    return;
                }
                if (Controller.ALWAYS_DECODE_ETH) {
                    eth = new Ethernet();
                    eth.deserialize(pi.getData(), 0, pi.getData().length);
                }
            default:
                List<IOFMessageListener> listeners = null;
                if (messageListeners.containsKey(m.getType())) {
                    listeners = messageListeners.get(m.getType()).getOrderedListeners();
                }
                FloodlightContext bc = null;
                if (listeners != null) {
                    if (bContext == null) {
                        bc = flcontext_alloc();
                    } else {
                        bc = bContext;
                    }
                    if (eth != null) {
                        IFloodlightProviderService.bcStore.put(bc,
                                IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                                eth);
                    }
                    pktinProcTimeService.bootstrap(listeners);
                    pktinProcTimeService.recordStartTimePktIn();
                    Command cmd;
                    for (IOFMessageListener listener : listeners) {
                        pktinProcTimeService.recordStartTimeComp(listener);
                        cmd = listener.receive(sw, m, bc);
                        pktinProcTimeService.recordEndTimeComp(listener);
                        if (Command.STOP.equals(cmd)) {
                            break;
                        }
                    }
                    pktinProcTimeService.recordEndTimePktIn(sw, m, bc);
                } else {
                    if (m.getType() != OFType.BARRIER_REPLY)
                        log.warn("Unhandled OF Message: {} from {}", m, sw);
                    else
                        log.debug("Received a Barrier Reply, no listeners for it");
                }
                for (IControllerCompletionListener listener:completionListeners)
                	listener.onMessageConsumed(sw, m, bc);
                if ((bContext == null) && (bc != null)) flcontext_free(bc);
        }
    }
    void reassertRole(OFSwitchHandshakeHandler ofSwitchHandshakeHandler, HARole role) {
        roleManager.reassertRole(ofSwitchHandshakeHandler, role);
    }
    @Override
    public String getControllerId() {
        return controllerId;
    }
    @Override
    public Set<IPv4Address> getOFAddresses() {
        return Collections.unmodifiableSet(openFlowAddresses);
    }
    @Override
    public TransportPort getOFPort() {
        return openFlowPort;
    }
    @Override
    public synchronized void addCompletionListener(IControllerCompletionListener listener) {
    	completionListeners.add(listener);
    }
    @Override
    public synchronized void removeCompletionListener(IControllerCompletionListener listener) {
    	String listenerName = listener.getName();
    	if (completionListeners.remove(listener)) {
    		log.debug("Removing completion listener {}" , listenerName);
    	} else {
    		log.warn("Trying to remove unknown completion listener {}" , listenerName);
    	}
    }
    @Override
    public synchronized void addOFMessageListener(OFType type, IOFMessageListener listener) {
        ListenerDispatcher<OFType, IOFMessageListener> ldd =
            messageListeners.get(type);
        if (ldd == null) {
            ldd = new ListenerDispatcher<OFType, IOFMessageListener>();
            messageListeners.put(type, ldd);
        }
        ldd.addListener(type, listener);
    }
    @Override
    public synchronized void removeOFMessageListener(OFType type, IOFMessageListener listener) {
        ListenerDispatcher<OFType, IOFMessageListener> ldd =
            messageListeners.get(type);
        if (ldd != null) {
            ldd.removeListener(listener);
        }
    }
    private void logListeners() {
        for (Map.Entry<OFType, ListenerDispatcher<OFType, IOFMessageListener>> entry : messageListeners.entrySet()) {
            OFType type = entry.getKey();
            ListenerDispatcher<OFType, IOFMessageListener> ldd = entry.getValue();
            StringBuilder sb = new StringBuilder();
            sb.append("OFListeners for ");
            sb.append(type);
            sb.append(": ");
            for (IOFMessageListener l : ldd.getOrderedListeners()) {
                sb.append(l.getName());
                sb.append(",");
            }
            log.debug(sb.toString());
        }
        StringBuilder sb = new StringBuilder();
        sb.append("HAListeners: ");
        for (IHAListener l: haListeners.getOrderedListeners()) {
            sb.append(l.getName());
            sb.append(", ");
        }
        log.debug(sb.toString());
    }
    public void removeOFMessageListeners(OFType type) {
        messageListeners.remove(type);
    }
    @Override
    public Map<OFType, List<IOFMessageListener>> getListeners() {
        Map<OFType, List<IOFMessageListener>> lers =
            new HashMap<OFType, List<IOFMessageListener>>();
        for(Entry<OFType, ListenerDispatcher<OFType, IOFMessageListener>> e : messageListeners.entrySet()) {
            lers.put(e.getKey(), e.getValue().getOrderedListeners());
        }
        return Collections.unmodifiableMap(lers);
    }
    @Override
    public void handleOutgoingMessage(IOFSwitch sw, OFMessage m) {
        if (sw == null)
            throw new NullPointerException("Switch must not be null");
        if (m == null)
            throw new NullPointerException("OFMessage must not be null");
        FloodlightContext bc = new FloodlightContext();
        List<IOFMessageListener> listeners = null;
        if (messageListeners.containsKey(m.getType())) {
            listeners = messageListeners.get(m.getType()).getOrderedListeners();
        }
        if (listeners != null) {
            for (IOFMessageListener listener : listeners) {
                if (Command.STOP.equals(listener.receive(sw, m, bc))) {
                    break;
                }
            }
        }
    }
    protected HARole getInitialRole(Map<String, String> configParams) {
        HARole role = HARole.STANDBY;
        String roleString = configParams.get("role");
        if (roleString != null) {
            try {
                role = HARole.valueOfBackwardsCompatible(roleString);
            }
            catch (IllegalArgumentException exc) {
                log.error("Invalid current role value: {}", roleString);
            }
        }
        log.info("Controller role set to {}", role);
        return role;
    }
    @Override
    public void run() {
        this.moduleLoaderState = ModuleLoaderState.COMPLETE;
        if (log.isDebugEnabled()) {
            logListeners();
        }
        while (true) {
            try {
                IUpdate update = updates.take();
                update.dispatch();
            } catch (InterruptedException e) {
                log.error("Received interrupted exception in updates loop;" +
                          "terminating process");
                log.info("Calling System.exit");
                System.exit(1);
            } catch (StorageException e) {
                log.error("Storage exception in controller " +
                          "updates loop; terminating process", e);
                log.info("Calling System.exit");
                System.exit(1);
            } catch (Exception e) {
                log.error("Exception in controller updates loop", e);
            }
        }
    }
    private void setConfigParams(Map<String, String> configParams) throws FloodlightModuleException {
        String ofPort = configParams.get("openFlowPort");
        if (!Strings.isNullOrEmpty(ofPort)) {
            try {
                openFlowPort = TransportPort.of(Integer.parseInt(ofPort));
            } catch (Exception e) {
                log.error("Invalid OpenFlow port {}, {}", ofPort, e);
                throw new FloodlightModuleException("Invalid OpenFlow port of " + ofPort + " in config");
            }
        }
        log.info("OpenFlow port set to {}", openFlowPort);
        String threads = configParams.get("workerThreads");
        if (!Strings.isNullOrEmpty(threads)) {
            this.workerThreads = Integer.parseInt(threads);
        }
        log.info("Number of worker threads set to {}", this.workerThreads);
        String addresses = configParams.get("openFlowAddresses");
        if (!Strings.isNullOrEmpty(addresses)) {
            try {
            } catch (Exception e) {
                log.error("Invalid OpenFlow address {}, {}", addresses, e);
                throw new FloodlightModuleException("Invalid OpenFlow address of " + addresses + " in config");
            }
            log.info("OpenFlow addresses set to {}", openFlowAddresses);
        } else {
        	openFlowAddresses.add(IPv4Address.NONE);
        }
    }
    public void init(Map<String, String> configParams) throws FloodlightModuleException {
        this.moduleLoaderState = ModuleLoaderState.INIT;
        this.messageListeners = new ConcurrentHashMap<OFType, ListenerDispatcher<OFType, IOFMessageListener>>();
        this.haListeners = new ListenerDispatcher<HAListenerTypeMarker, IHAListener>();
        this.controllerNodeIPsCache = new HashMap<String, String>();
        this.updates = new LinkedBlockingQueue<IUpdate>();
        this.providerMap = new HashMap<String, List<IInfoProvider>>();
        this.completionListeners = new ConcurrentLinkedQueue<IControllerCompletionListener>();
        setConfigParams(configParams);
        HARole initialRole = getInitialRole(configParams);
        this.notifiedRole = initialRole;
        this.shutdownService = new ShutdownServiceImpl();
        this.roleManager = new RoleManager(this, this.shutdownService,
                                           this.notifiedRole,
                                           INITIAL_ROLE_CHANGE_DESCRIPTION);
        this.timer = new HashedWheelTimer();
        this.switchService.registerLogicalOFMessageCategory(LogicalOFMessageCategory.MAIN);
        this.switchService.addOFSwitchListener(new NotificationSwitchListener());
        this.counters = new ControllerCounters(debugCounterService);
     }
    public void startupComponents(FloodlightModuleLoader floodlightModuleLoader) throws FloodlightModuleException {
        this.moduleLoaderState = ModuleLoaderState.STARTUP;
        storageSourceService.createTable(CONTROLLER_TABLE_NAME, null);
        storageSourceService.createTable(CONTROLLER_INTERFACE_TABLE_NAME, null);
        storageSourceService.createTable(SWITCH_CONFIG_TABLE_NAME, null);
        storageSourceService.setTablePrimaryKeyName(CONTROLLER_TABLE_NAME, CONTROLLER_ID);
        storageSourceService.addListener(CONTROLLER_INTERFACE_TABLE_NAME, this);
        storageSourceService.createTable(FLOW_PRIORITY_TABLE_NAME, null);
        storageSourceService.setTablePrimaryKeyName(FLOW_PRIORITY_TABLE_NAME, FLOW_COLUMN_PRIMARY_KEY);
        storageSourceService.addListener(FLOW_PRIORITY_TABLE_NAME, this);
        if (overload_drop) {
            this.loadmonitor.startMonitoring(this.threadPoolService.getScheduledExecutor());
        }
        restApiService.addRestletRoutable(new CoreWebRoutable());
        try {
            this.syncService.registerStore(OFSwitchManager.SWITCH_SYNC_STORE_NAME, Scope.LOCAL);
        } catch (SyncException e) {
            throw new FloodlightModuleException("Error while setting up sync service", e);
        }
        addInfoProvider("summary", this);
    }
    private void readFlowPriorityConfigurationFromStorage() {
        try {
            Map<String, Object> row;
            IResultSet resultSet = storageSourceService.executeQuery(
                FLOW_PRIORITY_TABLE_NAME, FLOW_COLUMN_NAMES, null, null);
            if (resultSet == null)
                return;
            for (Iterator<IResultSet> it = resultSet.iterator(); it.hasNext();) {
                row = it.next().getRow();
                if (row.containsKey(FLOW_COLUMN_PRIMARY_KEY)) {
                    String primary_key = (String) row.get(FLOW_COLUMN_PRIMARY_KEY);
                    if (primary_key.equals(FLOW_VALUE_PRIMARY_KEY)) {
                        if (row.containsKey(FLOW_COLUMN_ACCESS_PRIORITY)) {
                        }
                        if (row.containsKey(FLOW_COLUMN_CORE_PRIORITY)) {
                        }
                    }
                }
            }
        }
        catch (StorageException e) {
            log.error("Failed to access storage for forwarding configuration: {}", e.getMessage());
        }
        catch (NumberFormatException e) {
            log.error("Failed to read core or access flow priority from " +
                      "storage. Illegal number: {}", e.getMessage());
        }
    }
    @Override
    public void addInfoProvider(String type, IInfoProvider provider) {
        if (!providerMap.containsKey(type)) {
            providerMap.put(type, new ArrayList<IInfoProvider>());
        }
        providerMap.get(type).add(provider);
    }
    @Override
    public void removeInfoProvider(String type, IInfoProvider provider) {
        if (!providerMap.containsKey(type)) {
            log.debug("Provider type {} doesn't exist.", type);
            return;
        }
        providerMap.get(type).remove(provider);
    }
    @Override
    public Map<String, Object> getControllerInfo(String type) {
        if (!providerMap.containsKey(type)) return null;
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        for (IInfoProvider provider : providerMap.get(type)) {
            result.putAll(provider.getInfo(type));
        }
        return result;
    }
    @Override
    public void addHAListener(IHAListener listener) {
        this.haListeners.addListener(null,listener);
    }
    @Override
    public void removeHAListener(IHAListener listener) {
        this.haListeners.removeListener(listener);
    }
    protected void handleControllerNodeIPChanges() {
        HashMap<String,String> curControllerNodeIPs = new HashMap<String,String>();
        HashMap<String,String> addedControllerNodeIPs = new HashMap<String,String>();
        HashMap<String,String> removedControllerNodeIPs =new HashMap<String,String>();
        String[] colNames = { CONTROLLER_INTERFACE_CONTROLLER_ID,
                           CONTROLLER_INTERFACE_TYPE,
                           CONTROLLER_INTERFACE_NUMBER,
                           CONTROLLER_INTERFACE_DISCOVERED_IP };
        synchronized(controllerNodeIPsCache) {
            IResultSet res = storageSourceService.executeQuery(CONTROLLER_INTERFACE_TABLE_NAME,
                    colNames,null, null);
            while (res.next()) {
                if (res.getString(CONTROLLER_INTERFACE_TYPE).equals("Ethernet") &&
                        res.getInt(CONTROLLER_INTERFACE_NUMBER) == 0) {
                    String controllerID = res.getString(CONTROLLER_INTERFACE_CONTROLLER_ID);
                    String discoveredIP = res.getString(CONTROLLER_INTERFACE_DISCOVERED_IP);
                    String curIP = controllerNodeIPsCache.get(controllerID);
                    curControllerNodeIPs.put(controllerID, discoveredIP);
                    if (curIP == null) {
                        addedControllerNodeIPs.put(controllerID, discoveredIP);
                    }
                    else if (!curIP.equals(discoveredIP)) {
                        removedControllerNodeIPs.put(controllerID, curIP);
                        addedControllerNodeIPs.put(controllerID, discoveredIP);
                    }
                }
            }
            Set<String> curEntries = curControllerNodeIPs.keySet();
            Set<String> removedEntries = controllerNodeIPsCache.keySet();
            removedEntries.removeAll(curEntries);
            for (String removedControllerID : removedEntries)
                removedControllerNodeIPs.put(removedControllerID,
                                             controllerNodeIPsCache.get(removedControllerID));
            controllerNodeIPsCache.clear();
            controllerNodeIPsCache.putAll(curControllerNodeIPs);
            HAControllerNodeIPUpdate update = new HAControllerNodeIPUpdate(
                                curControllerNodeIPs, addedControllerNodeIPs,
                                removedControllerNodeIPs);
            if (!removedControllerNodeIPs.isEmpty() || !addedControllerNodeIPs.isEmpty()) {
                addUpdateToQueue(update);
            }
        }
    }
    @Override
    public Map<String, String> getControllerNodeIPs() {
        HashMap<String,String> retval = new HashMap<String,String>();
        synchronized(controllerNodeIPsCache) {
            retval.putAll(controllerNodeIPsCache);
        }
        return retval;
    }
    private static final String FLOW_PRIORITY_CHANGED_AFTER_STARTUP =
            "Flow priority configuration has changed after " +
            "controller startup. Restart controller for new " +
            "configuration to take effect.";
    @Override
    public void rowsModified(String tableName, Set<Object> rowKeys) {
        if (tableName.equals(CONTROLLER_INTERFACE_TABLE_NAME)) {
            handleControllerNodeIPChanges();
        } else if (tableName.equals(FLOW_PRIORITY_TABLE_NAME)) {
            log.warn(FLOW_PRIORITY_CHANGED_AFTER_STARTUP);
        }
    }
    @Override
    public void rowsDeleted(String tableName, Set<Object> rowKeys) {
        if (tableName.equals(CONTROLLER_INTERFACE_TABLE_NAME)) {
            handleControllerNodeIPChanges();
        } else if (tableName.equals(FLOW_PRIORITY_TABLE_NAME)) {
            log.warn(FLOW_PRIORITY_CHANGED_AFTER_STARTUP);
        }
    }
    @Override
    public long getSystemStartTime() {
        RuntimeMXBean rb = ManagementFactory.getRuntimeMXBean();
        return rb.getStartTime();
    }
    @Override
    public Map<String, Long> getMemory() {
        Map<String, Long> m = new HashMap<String, Long>();
        Runtime runtime = Runtime.getRuntime();
        m.put("total", runtime.totalMemory());
        m.put("free", runtime.freeMemory());
        return m;
    }
    @Override
    public Long getUptime() {
        RuntimeMXBean rb = ManagementFactory.getRuntimeMXBean();
        return rb.getUptime();
    }
    @Override
    public void addUpdateToQueue(IUpdate update) {
        try {
            this.updates.put(update);
        } catch (InterruptedException e) {
            log.error("Failure adding update {} to queue.", update);
        }
    }
    void processUpdateQueueForTesting() {
        while(!updates.isEmpty()) {
            IUpdate update = updates.poll();
            if (update != null)
                update.dispatch();
        }
    }
    boolean isUpdateQueueEmptyForTesting() {
        return this.updates.isEmpty();
    }
    void resetModuleState() {
        this.moduleLoaderState = ModuleLoaderState.INIT;
    }
    void setModuleLoaderStateForTesting(ModuleLoaderState state) {
        this.moduleLoaderState = state;
    }
    @Override
    public Map<String, Object> getInfo(String type) {
        if (!"summary".equals(type)) return null;
        Map<String, Object> info = new HashMap<String, Object>();
        info.put("# Switches", this.switchService.getAllSwitchDpids().size());
        return info;
    }
    protected void setNotifiedRole(HARole newRole) {
        notifiedRole = newRole;
    }
    @Override
    public RoleManager getRoleManager() {
        return this.roleManager;
    }
    public Optional<ControllerId> getId() {
        short nodeId = this.syncService.getLocalNodeId();
        if(nodeId == ClusterConfig.NODE_ID_UNCONFIGURED)
            return Optional.absent();
        else
            return Optional.of(ControllerId.of(nodeId));
    }
    @Override
    public Timer getTimer() {
        return this.timer;
    }
    public ControllerCounters getCounters() {
        return this.counters;
    }
}
