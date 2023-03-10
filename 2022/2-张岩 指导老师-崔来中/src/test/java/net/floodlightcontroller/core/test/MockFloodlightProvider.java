package net.floodlightcontroller.core.test;
import static org.junit.Assert.fail;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import io.netty.util.Timer;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.HAListenerTypeMarker;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IControllerCompletionListener;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IInfoProvider;
import net.floodlightcontroller.core.IListener.Command;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.RoleInfo;
import net.floodlightcontroller.core.internal.Controller.IUpdate;
import net.floodlightcontroller.core.internal.Controller.ModuleLoaderState;
import net.floodlightcontroller.core.internal.RoleManager;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.ListenerDispatcher;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.TransportPort;
import net.floodlightcontroller.packet.Ethernet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class MockFloodlightProvider implements IFloodlightModule, IFloodlightProviderService {
    private final static Logger log = LoggerFactory.getLogger(MockFloodlightProvider.class);
    protected ConcurrentMap<OFType, ListenerDispatcher<OFType,IOFMessageListener>> listeners;
    protected ListenerDispatcher<HAListenerTypeMarker, IHAListener> haListeners;
    private HARole role;
    private final Set<IPv4Address> openFlowHostname = Collections.singleton(IPv4Address.of("127.0.0.1"));
    private final TransportPort openFlowPort = TransportPort.of(6653);
    private final boolean useAsyncUpdates;
    private volatile ExecutorService executorService;
    private volatile Future<?> mostRecentUpdateFuture;
    private ConcurrentLinkedQueue<IControllerCompletionListener> completionListeners;
    public MockFloodlightProvider(boolean useAsyncUpdates) {
        listeners = new ConcurrentHashMap<OFType, ListenerDispatcher<OFType,
                                   IOFMessageListener>>();
        haListeners =
                new ListenerDispatcher<HAListenerTypeMarker, IHAListener>();
        completionListeners = 
        		new ConcurrentLinkedQueue<IControllerCompletionListener>();
        role = null;
        this.useAsyncUpdates = useAsyncUpdates;
    }
    public MockFloodlightProvider() {
        this(false);
    }
    @Override
    public synchronized void addOFMessageListener(OFType type,
                                                  IOFMessageListener listener) {
        ListenerDispatcher<OFType, IOFMessageListener> ldd =
                listeners.get(type);
        if (ldd == null) {
            ldd = new ListenerDispatcher<OFType, IOFMessageListener>();
            listeners.put(type, ldd);
        }
        ldd.addListener(type, listener);
    }
    @Override
    public synchronized void removeOFMessageListener(OFType type,
                                                     IOFMessageListener listener) {
        ListenerDispatcher<OFType, IOFMessageListener> ldd =
                listeners.get(type);
        if (ldd != null) {
            ldd.removeListener(listener);
        }
    }
    @Override
    public Map<OFType, List<IOFMessageListener>> getListeners() {
        Map<OFType, List<IOFMessageListener>> lers =
                new HashMap<OFType, List<IOFMessageListener>>();
        for(Entry<OFType, ListenerDispatcher<OFType, IOFMessageListener>> e :
            listeners.entrySet()) {
            lers.put(e.getKey(), e.getValue().getOrderedListeners());
        }
        return Collections.unmodifiableMap(lers);
    }
    public void clearListeners() {
        this.listeners.clear();
    }
    public void dispatchMessage(IOFSwitch sw, OFMessage msg) {
        dispatchMessage(sw, msg, new FloodlightContext());
    }
    public void dispatchMessage(IOFSwitch sw, OFMessage msg, FloodlightContext bc) {
        List<IOFMessageListener> theListeners = listeners.get(msg.getType()).getOrderedListeners();
        if (theListeners != null) {
            Command result = Command.CONTINUE;
            Iterator<IOFMessageListener> it = theListeners.iterator();
            if (OFType.PACKET_IN.equals(msg.getType())) {
                OFPacketIn pi = (OFPacketIn)msg;
                Ethernet eth = new Ethernet();
                eth.deserialize(pi.getData(), 0, pi.getData().length);
                IFloodlightProviderService.bcStore.put(bc,
                        IFloodlightProviderService.CONTEXT_PI_PAYLOAD,
                        eth);
            }
            while (it.hasNext() && !Command.STOP.equals(result)) {
                result = it.next().receive(sw, msg, bc);
            }
        }
        for (IControllerCompletionListener listener:completionListeners)
        	listener.onMessageConsumed(sw, msg, bc);
    }
    @Override
    public void handleOutgoingMessage(IOFSwitch sw, OFMessage m) {
        FloodlightContext bc = new FloodlightContext();
        List<IOFMessageListener> msgListeners = null;
        if (listeners.containsKey(m.getType())) {
            msgListeners = listeners.get(m.getType()).getOrderedListeners();
        }
        if (msgListeners != null) {
            for (IOFMessageListener listener : msgListeners) {
                if (Command.STOP.equals(listener.receive(sw, m, bc))) {
                    break;
                }
            }
        }
    }
    public void handleOutgoingMessages(IOFSwitch sw, List<OFMessage> msglist, FloodlightContext bc) {
        for (OFMessage m:msglist) {
            handleOutgoingMessage(sw, m);
        }
    }
    @Override
    public void run() {
        logListeners();
        if (useAsyncUpdates)
            executorService = Executors.newSingleThreadExecutor();
    }
    public void shutdown() {
        if (executorService != null) {
            executorService.shutdownNow();
            executorService = null;
            mostRecentUpdateFuture = null;
        }
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        Collection<Class<? extends IFloodlightService>> services =
                new ArrayList<Class<? extends IFloodlightService>>(1);
        services.add(IFloodlightProviderService.class);
        return services;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
            getServiceImpls() {
        Map<Class<? extends IFloodlightService>,
            IFloodlightService> m =
                new HashMap<Class<? extends IFloodlightService>,
                        IFloodlightService>();
        m.put(IFloodlightProviderService.class, this);
        return m;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
            getModuleDependencies() {
        return null;
    }
    @Override
    public void init(FloodlightModuleContext context) throws FloodlightModuleException {
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
    }
    @Override
    public void addInfoProvider(String type, IInfoProvider provider) {
    }
    @Override
    public void removeInfoProvider(String type, IInfoProvider provider) {
    }
    @Override
    public Map<String, Object> getControllerInfo(String type) {
        Map<String, Object> summary = new HashMap<String, Object>();
        summary.put("test-summary-1", 2);
        summary.put("test-summary-2", 5);
        return summary;
    }
    @Override
    public void addUpdateToQueue(final IUpdate update) {
        if (useAsyncUpdates) {
            mostRecentUpdateFuture = executorService.submit(new Runnable() {
                @Override
                public void run() {
                    update.dispatch();
                }
            });
        } else {
            update.dispatch();
        }
    }
    public void waitForUpdates(long timeout, TimeUnit unit) throws InterruptedException {
        long timeoutNanos = unit.toNanos(timeout);
        long start = System.nanoTime();
        for (;;) {
            Future<?> future = mostRecentUpdateFuture;
            if ((future == null) || future.isDone())
                break;
            Thread.sleep(100);
            long now = System.nanoTime();
            if (now > start + timeoutNanos) {
                fail("Timeout waiting for update tasks to complete");
            }
        }
    }
    @Override
    public void addHAListener(IHAListener listener) {
        haListeners.addListener(null,listener);
    }
    @Override
    public void removeHAListener(IHAListener listener) {
        haListeners.removeListener(listener);
    }
    @Override
    public HARole getRole() {
        if (this.role == null)
            throw new IllegalStateException("You need to call setRole on "
                       + "MockFloodlightProvider before calling startUp on "
                       + "other modules");
        return this.role;
    }
    @Override
    public void setRole(HARole role, String roleChangeDescription) {
        this.role = role;
    }
    public void transitionToActive() {
        IUpdate update = new IUpdate() {
            @Override
            public void dispatch() {
                for (IHAListener rl : haListeners.getOrderedListeners()) {
                    rl.transitionToActive();
                }
            }
        };
        addUpdateToQueue(update);
    }
    @Override
    public Map<String, String> getControllerNodeIPs() {
        return null;
    }
    @Override
    public long getSystemStartTime() {
        return 0;
    }
    private void logListeners() {
        for (Map.Entry<OFType,
                       ListenerDispatcher<OFType,
                                          IOFMessageListener>> entry
             : listeners.entrySet()) {
            OFType type = entry.getKey();
            ListenerDispatcher<OFType, IOFMessageListener> ldd =
                    entry.getValue();
            StringBuffer sb = new StringBuffer();
            sb.append("OFListeners for ");
            sb.append(type);
            sb.append(": ");
            for (IOFMessageListener l : ldd.getOrderedListeners()) {
                sb.append(l.getName());
                sb.append(",");
            }
            log.debug(sb.toString());
        }
    }
    @Override
    public RoleInfo getRoleInfo() {
        return null;
    }
    @Override
    public Map<String, Long> getMemory() {
        Map<String, Long> m = new HashMap<String, Long>();
        m.put("total", 1000000000L);
        m.put("free", 20000000L);
        return m;
    }
    @Override
    public Long getUptime() {
        return 1000000L;
    }
    @Override
    public Set<IPv4Address> getOFAddresses() {
        return openFlowHostname;
    }
    @Override
    public TransportPort getOFPort() {
        return openFlowPort;
    }
    @Override
    public void handleMessage(IOFSwitch sw, OFMessage m,
                              FloodlightContext bContext) {
    }
    @Override
    public Timer getTimer() {
        return null;
    }
    @Override
    public RoleManager getRoleManager() {
        return null;
    }
    @Override
    public ModuleLoaderState getModuleLoaderState() {
        return null;
    }
    @Override
    public String getControllerId() {
        return null;
    }
    @Override
    public Set<String> getUplinkPortPrefixSet() {
        return null;
    }
    @Override
    public int getWorkerThreads() {
        return 0;
    }
	@Override
	public void addCompletionListener(IControllerCompletionListener listener) {
		completionListeners.add(listener);
	}
	@Override
	public void removeCompletionListener(IControllerCompletionListener listener) {
		completionListeners.remove(listener);
	}
}
