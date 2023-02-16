package net.floodlightcontroller.virtualnetwork;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.AppCookie;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.IDeviceListener;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.packet.DHCP;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPacket;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.routing.ForwardingBase;
public class VirtualNetworkFilter
implements IFloodlightModule, IVirtualNetworkService, IOFMessageListener {
	protected static Logger log = LoggerFactory.getLogger(VirtualNetworkFilter.class);
	private static final short APP_ID = 20;
	static {
		AppCookie.registerApp(APP_ID, "VirtualNetworkFilter");
	}
	IFloodlightProviderService floodlightProviderService;
	IRestApiService restApiService;
	IDeviceService deviceService;
	protected DeviceListenerImpl deviceListener;
	protected void addGateway(String guid, IPv4Address ip) {
		if (ip.getInt() != 0) {
			if (log.isDebugEnabled()) {
				log.debug("Adding {} as gateway for GUID {}", ip.toString(), guid);
			}
			guidToGateway.put(guid, ip);
			if (vNetsByGuid.get(guid) != null)
				vNetsByGuid.get(guid).setGateway(ip.toString());
			if (gatewayToGuid.containsKey(ip)) {
				Set<String> gSet = gatewayToGuid.get(ip);
				gSet.add(guid);
			} else {
				Set<String> gSet = Collections.synchronizedSet(new HashSet<String>());
				gSet.add(guid);
				gatewayToGuid.put(ip, gSet);
			}
		}
	}
	protected void deleteGateway(String guid) {
		IPv4Address gwIp = guidToGateway.remove(guid);
		if (gwIp == null) return;
		Set<String> gSet = gatewayToGuid.get(gwIp);
		gSet.remove(guid);
		if (vNetsByGuid.get(guid) != null)
			vNetsByGuid.get(guid).setGateway(null);
	}
	@Override
	public void createNetwork(String guid, String network, IPv4Address gateway) {
		if (log.isDebugEnabled()) {
			String gw = null;
			try {
				gw = gateway.toString();
			} catch (Exception e) {
			}
			log.debug("Creating network {} with ID {} and gateway {}",
					new Object[] {network, guid, gw});
		}
		if (!nameToGuid.isEmpty()) {
			for (Entry<String, String> entry : nameToGuid.entrySet()) {
				if (entry.getValue().equals(guid)) {
					nameToGuid.remove(entry.getKey());
					break;
				}
			}
		}
		if(network != null)
			nameToGuid.put(network, guid);
		if (vNetsByGuid.containsKey(guid))
		else
		if ((gateway != null) && (gateway.getInt() != 0)) {
			addGateway(guid, gateway);
			if (vNetsByGuid.get(guid) != null)
				vNetsByGuid.get(guid).setGateway(gateway.toString());
		}
	}
	@Override
	public void deleteNetwork(String guid) {
		String name = null;
		if (nameToGuid.isEmpty()) {
			log.warn("Could not delete network with ID {}, network doesn't exist",
					guid);
			return;
		}
		for (Entry<String, String> entry : nameToGuid.entrySet()) {
			if (entry.getValue().equals(guid)) {
				name = entry.getKey();
				break;
			}
			log.warn("Could not delete network with ID {}, network doesn't exist", guid);
		}
		if (log.isDebugEnabled())
			log.debug("Deleting network with name {} ID {}", name, guid);
		nameToGuid.remove(name);
		deleteGateway(guid);
		if (vNetsByGuid.get(guid) != null){
			vNetsByGuid.get(guid).clearHosts();
			vNetsByGuid.remove(guid);
		}
		Collection<MacAddress> deleteList = new ArrayList<MacAddress>();
		for (MacAddress host : macToGuid.keySet()) {
			if (macToGuid.get(host).equals(guid)) {
				deleteList.add(host);
			}
		}
		for (MacAddress mac : deleteList) {
			if (log.isDebugEnabled()) {
				log.debug("Removing host {} from network {}", mac.toString(), guid);
			}
			macToGuid.remove(mac);
			for (Entry<String, MacAddress> entry : portToMac.entrySet()) {
				if (entry.getValue().equals(mac)) {
					portToMac.remove(entry.getKey());
					break;
				}
			}
		}
	}
	@Override
	public void addHost(MacAddress mac, String guid, String port) {
		if (guid != null) {
			if (log.isDebugEnabled()) {
				log.debug("Adding {} to network ID {} on port {}",
						new Object[] {mac, guid, port});
			}
			macToGuid.put(mac, guid);
			portToMac.put(port, mac);
			if (vNetsByGuid.get(guid) != null)
				vNetsByGuid.get(guid).addHost(port, mac);
		} else {
			log.warn("Could not add MAC {} to network ID {} on port {}, the network does not exist",
					new Object[] {mac.toString(), guid, port});
		}
	}
	@Override
	public void deleteHost(MacAddress mac, String port) {
		if (log.isDebugEnabled()) {
			log.debug("Removing host {} from port {}", mac, port);
		}
		if (mac == null && port == null) return;
		if (port != null) {
			MacAddress host = portToMac.remove(port);
			if (host != null && vNetsByGuid.get(macToGuid.get(host)) != null)
				vNetsByGuid.get(macToGuid.get(host)).removeHost(host);
			if (host != null)
				macToGuid.remove(host);
		} else if (mac != null) {
			if (!portToMac.isEmpty()) {
				for (Entry<String, MacAddress> entry : portToMac.entrySet()) {
					if (entry.getValue().equals(mac)) {
						if (vNetsByGuid.get(macToGuid.get(entry.getValue())) != null)
							vNetsByGuid.get(macToGuid.get(entry.getValue())).removeHost(entry.getValue());
						portToMac.remove(entry.getKey());
						macToGuid.remove(entry.getValue());
						return;
					}
				}
			}
		}
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IVirtualNetworkService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService>
	getServiceImpls() {
		Map<Class<? extends IFloodlightService>,
		IFloodlightService> m =
		new HashMap<Class<? extends IFloodlightService>,
		IFloodlightService>();
		m.put(IVirtualNetworkService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IRestApiService.class);
		l.add(IDeviceService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context)
			throws FloodlightModuleException {
		floodlightProviderService = context.getServiceImpl(IFloodlightProviderService.class);
		restApiService = context.getServiceImpl(IRestApiService.class);
		deviceService = context.getServiceImpl(IDeviceService.class);
		vNetsByGuid = new ConcurrentHashMap<String, VirtualNetwork>();
		nameToGuid = new ConcurrentHashMap<String, String>();
		guidToGateway = new ConcurrentHashMap<String, IPv4Address>();
		gatewayToGuid = new ConcurrentHashMap<IPv4Address, Set<String>>();
		macToGuid = new ConcurrentHashMap<MacAddress, String>();
		portToMac = new ConcurrentHashMap<String, MacAddress>();
		macToGateway = new ConcurrentHashMap<MacAddress, IPv4Address>();
		deviceListener = new DeviceListenerImpl();
	}
	@Override
	public void startUp(FloodlightModuleContext context) {
		floodlightProviderService.addOFMessageListener(OFType.PACKET_IN, this);
		restApiService.addRestletRoutable(new VirtualNetworkWebRoutable());
		deviceService.addListener(this.deviceListener);
	}
	@Override
	public String getName() {
		return "virtualizer";
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return (type.equals(OFType.PACKET_IN) &&
				(name.equals("linkdiscovery") || (name.equals("devicemanager"))));
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		return (type.equals(OFType.PACKET_IN) && name.equals("forwarding"));
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		switch (msg.getType()) {
		case PACKET_IN:
			return processPacketIn(sw, (OFPacketIn)msg, cntx);
		default:
			break;
		}
		log.warn("Received unexpected message {}", msg);
		return Command.CONTINUE;
	}
	protected boolean isDefaultGateway(Ethernet frame) {
		if (macToGateway.containsKey(frame.getSourceMACAddress()))
			return true;
		IPv4Address gwIp = macToGateway.get(frame.getDestinationMACAddress());
		if (gwIp != null) {
			MacAddress host = frame.getSourceMACAddress();
			String srcNet = macToGuid.get(host);
			if (srcNet != null) {
				IPv4Address gwIpSrcNet = guidToGateway.get(srcNet);
				if ((gwIpSrcNet != null) && (gwIp.equals(gwIpSrcNet)))
					return true;
			}
		}
		return false;
	}
	protected boolean oneSameNetwork(MacAddress m1, MacAddress m2) {
		String net1 = macToGuid.get(m1);
		String net2 = macToGuid.get(m2);
		if (net1 == null) return false;
		if (net2 == null) return false;
		return net1.equals(net2);
	}
	protected boolean isDhcpPacket(Ethernet frame) {
		if (payload == null) return false;
		if (p2 == null) return false;
		if ((p3 != null) && (p3 instanceof DHCP)) return true;
		return false;
	}
	protected Command processPacketIn(IOFSwitch sw, OFPacketIn msg, FloodlightContext cntx) {
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx,
				IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
		Command ret = Command.STOP;
		String srcNetwork = macToGuid.get(eth.getSourceMACAddress());
		if (eth.isBroadcast() || eth.isMulticast() || isDefaultGateway(eth) || isDhcpPacket(eth)) {
			ret = Command.CONTINUE;
		} else if (srcNetwork == null) {
			log.trace("Blocking traffic from host {} because it is not attached to any network.",
					eth.getSourceMACAddress().toString());
			ret = Command.STOP;
		} else if (oneSameNetwork(eth.getSourceMACAddress(), eth.getDestinationMACAddress())) {
			ret = Command.CONTINUE;
		}
		if (log.isTraceEnabled())
			log.trace("Results for flow between {} and {} is {}",
					new Object[] {eth.getSourceMACAddress(), eth.getDestinationMACAddress(), ret});
        if (ret == Command.STOP) {
            if (!(eth.getPayload() instanceof ARP))
                doDropFlow(sw, msg, cntx);
        }
		return ret;
	}
	protected void doDropFlow(IOFSwitch sw, OFPacketIn pi, FloodlightContext cntx) {
		if (log.isTraceEnabled()) {
			log.trace("doDropFlow pi={} srcSwitch={}",
					new Object[] { pi, sw });
		}
		if (sw == null) {
			log.warn("Switch is null, not installing drop flowmod for PacketIn {}", pi);
			return;
		}
		OFFlowMod.Builder fmb = sw.getOFFactory().buildFlowModify();
		U64 cookie = AppCookie.makeCookie(APP_ID, 0);
		fmb.setCookie(cookie)
		.setIdleTimeout(ForwardingBase.FLOWMOD_DEFAULT_IDLE_TIMEOUT)
		.setHardTimeout(ForwardingBase.FLOWMOD_DEFAULT_HARD_TIMEOUT)
		.setBufferId(OFBufferId.NO_BUFFER)
		.setMatch(pi.getMatch())
		.setActions(actions);
		if (log.isTraceEnabled()) {
			log.trace("write drop flow-mod srcSwitch={} match={} " +
					"pi={} flow-mod={}",
					new Object[] {sw, pi.getMatch(), pi, fmb.build()});
		}
		sw.write(fmb.build());
		return;
	}
	@Override
	public Collection <VirtualNetwork> listNetworks() {
		return vNetsByGuid.values();
	}
	class DeviceListenerImpl implements IDeviceListener{
		@Override
		public void deviceAdded(IDevice device) {
			if (device.getIPv4Addresses() == null) return;
			for (IPv4Address i : device.getIPv4Addresses()) {
				if (gatewayToGuid.containsKey(i)) {
					MacAddress mac = device.getMACAddress();
					if (log.isDebugEnabled())
						log.debug("Adding MAC {} with IP {} a a gateway",
								mac.toString(),
								i.toString());
					macToGateway.put(mac, i);
				}
			}
		}
		@Override
		public void deviceRemoved(IDevice device) {
			MacAddress mac = device.getMACAddress();
			if (macToGateway.containsKey(mac)) {
				if (log.isDebugEnabled())
					log.debug("Removing MAC {} as a gateway", mac.toString());
				macToGateway.remove(mac);
			}
		}
		@Override
		public void deviceIPV4AddrChanged(IDevice device) {
			deviceAdded(device);
		}
		@Override
		public void deviceIPV6AddrChanged(IDevice device) {
			log.debug("IPv6 address change not handled in VirtualNetworkFilter. Device: {}", device.toString());
		}
		@Override
		public void deviceMoved(IDevice device) {
		}
		@Override
		public void deviceVlanChanged(IDevice device) {
		}
		@Override
		public String getName() {
			return VirtualNetworkFilter.this.getName();
		}
		@Override
		public boolean isCallbackOrderingPrereq(String type, String name) {
			return false;
		}
		@Override
		public boolean isCallbackOrderingPostreq(String type, String name) {
			return false;
		}
	}
}
