package net.floodlightcontroller.core;
import java.util.List;
import java.util.Set;
import java.util.Map;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.packet.Ethernet;
import io.netty.util.Timer;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IInfoProvider;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.RoleInfo;
import net.floodlightcontroller.core.internal.RoleManager;
import net.floodlightcontroller.core.internal.Controller.IUpdate;
import net.floodlightcontroller.core.internal.Controller.ModuleLoaderState;
import net.floodlightcontroller.core.FloodlightContextStore;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.TransportPort;
public interface IFloodlightProviderService extends
        IFloodlightService, Runnable {
    public static final String CONTEXT_PI_PAYLOAD =
            "net.floodlightcontroller.core.IFloodlightProvider.piPayload";
    public static final FloodlightContextStore<Ethernet> bcStore =
            new FloodlightContextStore<Ethernet>();
    public static final String SERVICE_DIRECTORY_SERVICE_NAME = "openflow";
    public void addOFMessageListener(OFType type, IOFMessageListener listener);
    public void removeOFMessageListener(OFType type, IOFMessageListener listener);
    public Map<OFType, List<IOFMessageListener>> getListeners();
    public HARole getRole();
    public RoleInfo getRoleInfo();
    public Map<String,String> getControllerNodeIPs();
    public String getControllerId();
    public Set<IPv4Address> getOFAddresses();
    public TransportPort getOFPort();
    public void setRole(HARole role, String changeDescription);
    public void addUpdateToQueue(IUpdate update);
    public void addHAListener(IHAListener listener);
    public void removeHAListener(IHAListener listener);
    public void handleOutgoingMessage(IOFSwitch sw, OFMessage m);
    @Override
    public void run();
    public void addInfoProvider(String type, IInfoProvider provider);
   public void removeInfoProvider(String type, IInfoProvider provider);
   public Map<String, Object> getControllerInfo(String type);
   public long getSystemStartTime();
   public Map<String, Long> getMemory();
   public Long getUptime();
   public Set<String> getUplinkPortPrefixSet();
   public void handleMessage(IOFSwitch sw, OFMessage m,
                          FloodlightContext bContext);
   public Timer getTimer();
   public RoleManager getRoleManager();
   ModuleLoaderState getModuleLoaderState();
   public int getWorkerThreads();
   void addCompletionListener(IControllerCompletionListener listener);
   void removeCompletionListener(IControllerCompletionListener listener);
}
