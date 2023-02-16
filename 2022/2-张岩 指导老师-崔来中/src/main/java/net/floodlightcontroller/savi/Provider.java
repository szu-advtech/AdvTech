package net.floodlightcontroller.savi;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFlowAdd;
import org.projectfloodlight.openflow.protocol.OFFlowDelete;
import org.projectfloodlight.openflow.protocol.OFFlowDeleteStrict;
import org.projectfloodlight.openflow.protocol.OFFlowModify;
import org.projectfloodlight.openflow.protocol.OFFlowModifyStrict;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFMeterMod;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketOut;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.action.OFActionOutput;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
import org.projectfloodlight.openflow.protocol.instruction.OFInstructionApplyActions;
import org.projectfloodlight.openflow.protocol.instruction.OFInstructionMeter;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.protocol.meterband.OFMeterBand;
import org.projectfloodlight.openflow.protocol.meterband.OFMeterBandDrop;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitchListener;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.AppCookie;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.linkdiscovery.ILinkDiscoveryListener;
import net.floodlightcontroller.linkdiscovery.ILinkDiscoveryService;
import net.floodlightcontroller.packet.ARP;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.IPv6;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.routing.IRoutingDecision;
import net.floodlightcontroller.routing.IRoutingDecision.RoutingAction;
import net.floodlightcontroller.routing.RoutingDecision;
import net.floodlightcontroller.savi.action.Action;
import net.floodlightcontroller.savi.action.Action.ActionFactory;
import net.floodlightcontroller.savi.action.BindIPv4Action;
import net.floodlightcontroller.savi.action.BindIPv6Action;
import net.floodlightcontroller.savi.action.CheckIPv4BindingAction;
import net.floodlightcontroller.savi.action.CheckIPv6BindingAction;
import net.floodlightcontroller.savi.action.FloodAction;
import net.floodlightcontroller.savi.action.PacketOutAction;
import net.floodlightcontroller.savi.action.UnbindIPv4Action;
import net.floodlightcontroller.savi.action.UnbindIPv6Action;
import net.floodlightcontroller.savi.binding.Binding;
import net.floodlightcontroller.savi.binding.BindingManager;
import net.floodlightcontroller.savi.flow.FlowAction;
import net.floodlightcontroller.savi.flow.FlowAddAction;
import net.floodlightcontroller.savi.flow.FlowModAction;
import net.floodlightcontroller.savi.flow.FlowRemoveAction;
import net.floodlightcontroller.savi.rest.SAVIRestRoute;
import net.floodlightcontroller.savi.service.SAVIProviderService;
import net.floodlightcontroller.savi.service.SAVIService;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import net.floodlightcontroller.topology.ITopologyListener;
import net.floodlightcontroller.topology.ITopologyService;
import sun.nio.cs.ext.TIS_620;
import javax.crypto.Mac;
public class Provider implements IFloodlightModule, IOFSwitchListener, 
IOFMessageListener, ITopologyListener, SAVIProviderService, ILinkDiscoveryListener{
	static final int PROTOCOL_LAYER_PRIORITY = 1;
	static final int SERVICE_LAYER_PRIORITY = 7;
	static final int BINDING_LAYER_PRIORITY = 5;
	static final int RELIABLE_PORT_PRIORITY = 200;
	static final long BAND_RATE = 25;
	static final Logger log = LoggerFactory.getLogger(SAVIProviderService.class);
	protected boolean ENABLE_METER_TABLE = true;
	protected IFloodlightProviderService floodlightProvider;
	protected IOFSwitchService switchService;
	protected IDeviceService deviceService;
	protected ITopologyService topologyService;
	protected IRestApiService restApiService;
	protected IThreadPoolService threadPoolService;
	protected ILinkDiscoveryService linkDiscoveryService;
	protected SingletonTask updateTask;
	protected SingletonTask printCount;
	protected List<SAVIService> saviServices;
	protected BindingManager manager;
	protected List<Match> serviceRules;
	protected List<Match> protocolRules;
	protected Set<SwitchPort> securityPort;
	protected Queue<LDUpdate> updateQueue;
	protected static final boolean ENABLE_FAST_FLOOD = true;
	public static final int SAVI_PROVIDER_APP_ID = 1000;
	public static TableId STATIC_TABLE_ID=TableId.of(0);
	public static TableId DYNAMIC_TABLE_ID=TableId.of(1);
	public static TableId FLOW_TABLE_ID = TableId.of(2);
	public static int securityTableCounter = 0;
	private int updateTime;
	private int hardTimeout;
	public static final int STATIC_FITST_PRIORITY=11111;
	private Set<SwitchPort> pushFlowToSwitchPorts=new HashSet<>();
	private Set<SwitchPort> edgeSwitch=new HashSet<>();
	volatile int packetCount=0;
	volatile int preCount;
	File file;
	BufferedWriter bw;
	static {
		AppCookie.registerApp(SAVI_PROVIDER_APP_ID, "Forwarding");
	}
	public static final U64 cookie = AppCookie.makeCookie(SAVI_PROVIDER_APP_ID, 0);
	protected Command processPacketIn(IOFSwitch sw, OFPacketIn pi, FloodlightContext cntx) {
		OFPort inPort = (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort()
				: pi.getMatch().get(MatchField.IN_PORT));
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx, IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
		IRoutingDecision decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
		if(dstDevice == null) {
			log.info("no dstdevice");
		SwitchPort switchPort = new SwitchPort(sw.getId(), inPort);
		RoutingAction routingAction = null;
		if (decision == null) {
			decision = new RoutingDecision(sw.getId(), inPort,
					IDeviceService.fcStore.get(cntx, IDeviceService.CONTEXT_SRC_DEVICE), RoutingAction.FORWARD);
		}
		for(SAVIService s : saviServices) {
			if (s.match(eth)) {
				routingAction = s.process(switchPort, eth);
				break;
			}
		}
		if(routingAction == null) {
			routingAction = process(switchPort, eth);
		}
		if(routingAction != null) {
			decision.setRoutingAction(routingAction);
		}
		decision.addToContext(cntx);
		return Command.CONTINUE;
	}
	@Override
	public void addSAVIService(SAVIService service) {
		saviServices.add(service);
		serviceRules.addAll(service.getMatches());
	}
	@Override
	public boolean pushActions(List<Action> actions) {
		for(Action action:actions){
			switch(action.getType()){
			case FLOOD:
				doFlood((FloodAction)action);
				break;
			case PACKET_OUT:
			case PACKET_OUT_MULTI_PORT:
				doPacketOut((PacketOutAction)action);
				break;
			case BIND_IPv4:
				doBindIPv4((BindIPv4Action)action);
				break;
			case BIND_IPv6:
				doBindIPv6((BindIPv6Action)action);
				break;
			case UNBIND_IPv4:
				doUnbindIPv4((UnbindIPv4Action)action);
				break;
			case UNBIND_IPv6:
				doUnbindIPv6((UnbindIPv6Action)action);
				break;
			case CHECK_IPv4_BINDING:
				return doCheckIPv4BInding((CheckIPv4BindingAction)action);
			case CHECK_IPv6_BINDING:
				return doCheckIPv6Binding((CheckIPv6BindingAction)action);
			default:
				break;
			}
		}
		return true;
	}
	public boolean pushFlowActions(List<FlowAction> actions){
		for(FlowAction action:actions){
			switch(action.getType()){
			case ADD:
				doFlowAdd((FlowAddAction)action);
				break;
			case MOD:
				doFlowMod((FlowModAction)action);
				break;
			case REMOVE:
				doFlowRemove((FlowRemoveAction)action);
				break;
			default:
				break;
			}
		}
		return true;
	}
	private void AddTimingFlowEntry(int hardTimeout, DatapathId dpid) {
		Match.Builder mb=OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		List<OFInstruction> instructions=new ArrayList<>();
		instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(DYNAMIC_TABLE_ID));
		doFlowAdd(dpid, STATIC_TABLE_ID, mb.build(), null, instructions, STATIC_FITST_PRIORITY, hardTimeout, 0);
	}
	@Override
	public void convertTable(boolean isTrue) {
		for(DatapathId dpid : portsInBind.keySet()) {
			convertTable(dpid,isTrue);
		}
	}
	@Override
	public void convertTable(DatapathId dpid, boolean isTrue) {
		if(!portsInBind.containsKey(dpid)) {
			log.warn("交换机 "+dpid+" 不是边缘交换机，不用转换");
			return ;
		}
		Match.Builder mb=OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		if(isTrue) {
			if(staticSwId.contains(dpid)) {
				return ;
			}
			staticSwId.add(dpid);
			doFlowRemove(dpid, STATIC_TABLE_ID, mb.build(), STATIC_FITST_PRIORITY);
			log.warn("交换机{"+dpid+"}转为静态流表");
		}else {
			if(!staticSwId.contains(dpid)) {
				return ;
			}
			List<OFInstruction> instructions=new ArrayList<>();
			instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
			doFlowAdd(dpid, STATIC_TABLE_ID, mb.build(), null, instructions, STATIC_FITST_PRIORITY);
			staticSwId.remove(dpid);
			log.warn("交换机{"+dpid+"}转为动态流表");
		}
	}
	@Override
	public String getName() {
		return "savi";
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg,
			FloodlightContext cntx) {
		switch (msg.getType()) {
		case PACKET_IN:
			packetCount++;
			return processPacketIn(sw, (OFPacketIn) msg, cntx);
		case ERROR:
			log.info("ERROR");
		default:
			break;
		}
		return Command.CONTINUE;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> services = new ArrayList<Class<? extends IFloodlightService>>();
		services.add(SAVIProviderService.class);
		return services;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		Map<Class<? extends IFloodlightService>, IFloodlightService> serviceImpls = new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
		serviceImpls.put(SAVIProviderService.class, this);
		return serviceImpls;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> dependencies = new ArrayList<Class<? extends IFloodlightService>>();
		dependencies.add(IFloodlightProviderService.class);
		dependencies.add(IOFSwitchService.class);
		dependencies.add(IDeviceService.class);
		dependencies.add(ITopologyService.class);
		dependencies.add(IStorageSourceService.class);
		dependencies.add(IRestApiService.class);
		dependencies.add(IThreadPoolService.class);
		dependencies.add(ILinkDiscoveryService.class);
		return dependencies;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProvider 	 = context.getServiceImpl(IFloodlightProviderService.class);
		switchService 	   	 = context.getServiceImpl(IOFSwitchService.class);
		deviceService 	   	 = context.getServiceImpl(IDeviceService.class);
		topologyService 	 = context.getServiceImpl(ITopologyService.class);
		restApiService 		 = context.getServiceImpl(IRestApiService.class);
		threadPoolService	 = context.getServiceImpl(IThreadPoolService.class);
		linkDiscoveryService = context.getServiceImpl(ILinkDiscoveryService.class);
		updateQueue = new ConcurrentLinkedQueue<>();
		saviServices 		= new ArrayList<>();
		manager 			= new BindingManager();
		serviceRules		= new ArrayList<>();
		protocolRules		= new ArrayList<>();
		updateTime=6;
		hardTimeout=18;
		staticSwId=new HashSet<>();
		portsInBind=new HashMap<>();
		rank=new HashMap<>();
		{
			Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
			mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
			protocolRules.add(mb.build());
			mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
			mb.setExact(MatchField.ETH_TYPE, EthType.IPv4);
			protocolRules.add(mb.build());
		}
		securityPort = new HashSet<>();
		{
		}
		Map<String, String> configParameters = context.getConfigParams(this);
		if(configParameters.containsKey("enable-meter-table")) {
			if(configParameters.get("enable-meter-table").equals("YES")) {
				ENABLE_METER_TABLE = true;
			}
			else {
				ENABLE_METER_TABLE = false;
			}
		}
		else {
			ENABLE_METER_TABLE = false;
		}
		initIO();
	} 
	@Override
	public void startUp(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProvider.addOFMessageListener(OFType.PACKET_IN, this);
		floodlightProvider.addOFMessageListener(OFType.ERROR, this);
		switchService.addOFSwitchListener(this);
		restApiService.addRestletRoutable(new SAVIRestRoute());
		linkDiscoveryService.addListener(this);
		ScheduledExecutorService ses = threadPoolService.getScheduledExecutor();
		updateTask = new SingletonTask(ses, new Runnable() {
			@Override
			public void run() {
				while(updateQueue.peek() != null){
					LDUpdate update = updateQueue.remove();
					switch(update.getOperation()){
					case PORT_UP:
						if(update.getSrc().getLong()>4||update.getSrcPort().getPortNumber()>2){
							securityPort.add(new SwitchPort(update.getSrc(), update.getSrcPort()));
							List<OFInstruction> instructions = new ArrayList<>();
							instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
							Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
							mb.setExact(MatchField.IN_PORT, update.getSrcPort());
							log.info("安全端口"+update.getSrc()+"--"+update.getSrcPort());
							doFlowAdd(update.getSrc(), STATIC_TABLE_ID, mb.build(), null, instructions, 1+BINDING_LAYER_PRIORITY);
						}
						if(update.getSrc().getLong()<5&&update.getSrcPort().getPortNumber()<3){
							System.out.println("=====Port_Up====");
							DatapathId dpid=update.getSrc();
							int port=update.getSrcPort().getPortNumber();
							addSpecialFlowEntry(dpid, port);
							if (ENABLE_METER_TABLE) {
								System.out.println("=======ENABLE_METER_TABLE======");
								List<OFAction> actions = new ArrayList<>();
								actions.add(OFFactories.getFactory(OFVersion.OF_14).actions().output(OFPort.CONTROLLER, Integer.MAX_VALUE));
								List<OFInstruction> instructions = new ArrayList<>();
								OFInstructionMeter meter = OFFactories.getFactory(OFVersion.OF_14).instructions().buildMeter()
										.setMeterId(port)
										.build();
								OFInstructionApplyActions output = OFFactories.getFactory(OFVersion.OF_14).instructions()
										.buildApplyActions().setActions(actions).build();
								instructions.add(meter);
								instructions.add(output);
								Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
								mb.setExact(MatchField.IN_PORT, update.getSrcPort());
								doFlowAdd(update.getSrc(), DYNAMIC_TABLE_ID, mb.build(), null, instructions, 2);
							}
						}
						break;
					case PORT_DOWN:
						SwitchPort sp=new SwitchPort(update.getSrc(), update.getSrcPort());
						for(SAVIService s : saviServices){
							s.handlePortDown(sp);
						}
						break;
					default:
						break;
					}
				}
				updateTask.reschedule(1000, TimeUnit.MILLISECONDS);
			}
		});
		updateTask.reschedule(100, TimeUnit.MILLISECONDS);
		ScheduledExecutorService ses0 = threadPoolService.getScheduledExecutor();
		printBindingTable=new SingletonTask(ses0, new Runnable() {
			@Override
			public void run() {
				System.out.println("========绑定表=========");
				List<Binding<?>> list=getBindings();
				StringBuilder sb=new StringBuilder();
				Map<Long, List<String>> printMap=new HashMap<>();
				if (list != null && !list.isEmpty()) {
					for(Binding<?> binding : list) {
						IPAddress<?> ip=binding.getAddress();
						if(ip.getIpVersion()== IPVersion.IPv6) {
							IPv6Address iPv6Address=(IPv6Address)ip;
							MacAddress macAddress=binding.getMacAddress();
							SwitchPort switchPort=binding.getSwitchPort();
							sb.append("SwitchPort：[s"+switchPort.getSwitchDPID().getLong()+", "+
									switchPort.getPort().getShortPortNumber()+"]; MAC:"+
									macAddress.toString()+"; IP:"+iPv6Address.toString());
							if (!printMap.containsKey(macAddress.getLong())) {
								List<String> temp=new ArrayList<>();
								printMap.put(macAddress.getLong(), temp);
							}
							printMap.get(macAddress.getLong()).add(sb.toString());
							sb.delete(0, sb.length());
						}
					}
					for(long i=1;i<9;i++){
						if(printMap.get(i)==null)
							continue;
						for(String str : printMap.get(i)){
							System.out.println(str);
						}
					}
				}
				printBindingTable.reschedule(5, TimeUnit.SECONDS);
			}
		});
		printBindingTable.reschedule(20, TimeUnit.SECONDS);
		ScheduledExecutorService ses1 = threadPoolService.getScheduledExecutor();
		printCount=new SingletonTask(ses1, new Runnable() {
			@Override
			public void run() {
				writeToTxt(packetCount-preCount);
				preCount=packetCount;
				printCount.reschedule(1, TimeUnit.SECONDS);
			}
		});
		printCount.reschedule(20, TimeUnit.SECONDS);
	}
	private void initIO(){
		file=new File(("savilog/count_d18.txt"));
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		try {
			bw=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, false)));
		} catch (FileNotFoundException e) {
			try {
				if(bw!=null)
					bw.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			e.printStackTrace();
		}
	}
	private void writeToTxt(int diff){
		try {
			bw.write(Integer.toString(diff));
			bw.newLine();
			bw.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	@Override
	public void switchAdded(DatapathId switchId) {
		manager.addSwitch(switchId);
		Match.Builder mb=OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		List<OFInstruction> instructions = new ArrayList<>();
		instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
		doFlowAdd(switchId, STATIC_TABLE_ID, mb.build(), null, null, 0);
		List<OFAction> actions = new ArrayList<>();
		actions.add(OFFactories.getFactory(OFVersion.OF_14).actions().output(OFPort.CONTROLLER, Integer.MAX_VALUE));
		doFlowAdd(switchId, FLOW_TABLE_ID, mb.build(), actions, null, 0);
	}
	private void addSpecialFlowEntry(DatapathId switchId, int meterId) {
		if(ENABLE_METER_TABLE) {
			OFMeterBandDrop.Builder bandBuilder = OFFactories.getFactory(OFVersion.OF_14)
					.meterBands().buildDrop().setRate(BAND_RATE);
			List<OFMeterBand> bands = new ArrayList<>();
			bands.add(bandBuilder.build());
			doMeterMod(switchId, meterId, bands);
		}
		Match.Builder mb=OFFactories.getFactory(OFVersion.OF_14).buildMatch();
			动态用来做端口分类 端口上收到的监听报文 根据端口分类 table-miss交给控制器 且不需要限速
		List<OFAction> actions = new ArrayList<>();
		actions.add(OFFactories.getFactory(OFVersion.OF_14).actions().output(OFPort.CONTROLLER, Integer.MAX_VALUE));
		doFlowAdd(switchId, DYNAMIC_TABLE_ID, mb.build(), actions, null, 0);
		for(Match match:serviceRules){
			List<OFInstruction> instructions = new ArrayList<>();
			instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(DYNAMIC_TABLE_ID));
			doFlowAdd(switchId, STATIC_TABLE_ID, match, null, instructions, SERVICE_LAYER_PRIORITY);
		}
		for(SwitchPort switchPort:securityPort){
			if(!switchId.equals(switchPort.getSwitchDPID())) continue;
			log.info("Provider line:750，reliable port " + switchPort.getSwitchDPID().toString() + "-" + switchPort.getPort().toString());
			mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
			mb.setExact(MatchField.IN_PORT, switchPort.getPort());
			List<OFInstruction> instructions = new ArrayList<>();
			instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
			doFlowAdd(switchPort.getSwitchDPID(), STATIC_TABLE_ID, mb.build(), null, instructions, BINDING_LAYER_PRIORITY);
		}
	}
	@Override
	public void switchRemoved(DatapathId switchId) {
		manager.removeSwitch(switchId);
		List<Action> actions = new ArrayList<>();
		actions.add(ActionFactory.getClearSwitchBindingAction(switchId));
		for(SAVIService s:saviServices){
			s.pushActins(actions);
		}
	}
	@Override
	public void switchActivated(DatapathId switchId) {
	}
	@Override
	public void switchPortChanged(DatapathId switchId, OFPortDesc port, PortChangeType type) {
	}
	@Override
	public void switchChanged(DatapathId switchId) {
	}
	protected RoutingAction process(SwitchPort switchPort, Ethernet eth){
		MacAddress macAddress = eth.getSourceMACAddress();
		if(securityPort.contains(switchPort) || !topologyService.isEdge(switchPort.getSwitchDPID(), switchPort.getPort())){
				return RoutingAction.FORWARD_OR_FLOOD;
		}
		if(eth.getEtherType() == EthType.IPv4){
			IPv4 ipv4 = (IPv4)eth.getPayload();
			IPv4Address address = ipv4.getSourceAddress();
			if(manager.check(switchPort, macAddress, address)){
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else if(address.isUnspecified()){
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else {
				return RoutingAction.NONE;
			}
		}
		else if(eth.getEtherType() == EthType.IPv6){
			IPv6 ipv6 = (IPv6)eth.getPayload();
			IPv6Address address = ipv6.getSourceAddress();
			if(address.isUnspecified()){
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else if(manager.check(switchPort, macAddress, address)){
				if(ipv6.getDestinationAddress().isBroadcast()
						||ipv6.getDestinationAddress().isMulticast()){
					return RoutingAction.MULTICAST;
				}
				else{
					return RoutingAction.FORWARD_OR_FLOOD;
				}
			}
			else{
				return RoutingAction.NONE;
			}
		}
		else if(eth.getEtherType() == EthType.ARP){
			ARP arp = (ARP)eth.getPayload();
			IPv4Address address = arp.getSenderProtocolAddress();
			if(manager.check(switchPort, macAddress, address)){
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else if(address.isUnspecified()){
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else {
				return RoutingAction.NONE;
			}
		}
		return null;
	}
	protected void doFlood(FloodAction action){
		SwitchPort inSwitchPort = new SwitchPort(action.getSwitchId(), action.getInPort());
		byte[] data = action.getEthernet().serialize();
		doFlood(inSwitchPort, data);
	}
	protected void doFlood(SwitchPort inSwitchPort, byte[] data){
		if(ENABLE_FAST_FLOOD) {
			doFastFlood(inSwitchPort, data);
			return;
		}
		Collection<? extends IDevice> tmp = deviceService.getAllDevices();
		for (IDevice d : tmp) {
			SwitchPort[] switchPorts = d.getAttachmentPoints();
			for (SwitchPort switchPort : switchPorts) {
				if (!switchPort.equals(inSwitchPort)) {
					doPacketOut(switchPort, data);
				}
			}
		}
	}
	protected void doFastFlood(SwitchPort inPort, byte[] data) {
		List<OFPort> ports = new ArrayList<>();
		IOFSwitch sw = switchService.getSwitch(inPort.getSwitchDPID());
		for(OFPort port: sw.getEnabledPortNumbers()) {
			if(!port.equals(inPort.getPort())&&topologyService.isEdge(sw.getId(), port)) {
				doPacketOut(new SwitchPort(inPort.getSwitchDPID(), port), data);
			}
		}
		for(DatapathId switchId: switchService.getAllSwitchDpids()) {
			if(!switchId.equals(inPort.getSwitchDPID())) {
				sw = switchService.getSwitch(switchId);
				ports.clear();
				for(OFPort port: sw.getEnabledPortNumbers()) {
					if(topologyService.isEdge(sw.getId(), port)) {
						doPacketOut(new SwitchPort(switchId, port), data);
					}
				}
			}
		}
	}
	protected void doPacketOut(PacketOutAction action) {
		doPacketOut(action.getSwitchId(),
					action.getInPort(),
					action.getOutPorts(),
					action.getEthernet().serialize());
	}
	protected void doPacketOut(SwitchPort switchPort, byte[] data) {
		IOFSwitch sw = switchService.getActiveSwitch(switchPort.getSwitchDPID());
		OFPort port = switchPort.getPort();
		try {
			OFPacketOut.Builder pob = sw.getOFFactory().buildPacketOut();
			List<OFAction> actions = new ArrayList<OFAction>();
			actions.add(sw.getOFFactory().actions().output(port, Integer.MAX_VALUE));
			pob.setActions(actions)
			   .setBufferId(OFBufferId.NO_BUFFER)
			   .setData(data)
			   .setInPort(OFPort.CONTROLLER);
			sw.write(pob.build());
		} catch (NullPointerException e) {
		}
	}
	protected void doPacketOut(DatapathId switchId, OFPort inPort, List<OFPort> outPorts, byte[] data) {
		IOFSwitch sw = switchService.getActiveSwitch(switchId);
		OFPacketOut.Builder pob = sw.getOFFactory().buildPacketOut();
		List<OFAction> actions = new ArrayList<OFAction>();
		for(OFPort port:outPorts) {
			actions.add(sw.getOFFactory().actions().output(port, Integer.MAX_VALUE));
		}
		pob.setActions(actions)
		   .setBufferId(OFBufferId.NO_BUFFER)
		   .setData(data)
		   .setInPort(inPort);
		sw.write(pob.build());
	}
	protected void doBindIPv4(BindIPv4Action action){
		Binding<?> binding = action.getBinding();
		log.info("BIND "+binding.getAddress());
		manager.addBinding(binding);
		if(securityPort.contains(binding.getSwitchPort())){
			return;
		}
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		mb.setExact(MatchField.ETH_SRC, binding.getMacAddress());
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv4);
		mb.setExact(MatchField.IPV4_SRC, (IPv4Address)binding.getAddress());
		mb.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
		List<OFInstruction> instructions = new ArrayList<>();
		instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
		doFlowAdd(binding.getSwitchPort().getSwitchDPID(), STATIC_TABLE_ID, mb.build(), null, instructions, BINDING_LAYER_PRIORITY);	
	}
	protected void doBindIPv6(BindIPv6Action action){
		Binding<?> binding = action.getBinding();
		log.info("BIND "+binding.getAddress().toString()+"  "+binding.getSwitchPort().getSwitchDPID());
		manager.addBinding(binding);
		if(securityPort.contains(binding.getSwitchPort())){
			return;
		}
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		if(!portsInBind.containsKey(binding.getSwitchPort().getSwitchDPID())){
			portsInBind.put(binding.getSwitchPort().getSwitchDPID(), 1);
		}else {
			portsInBind.put(binding.getSwitchPort().getSwitchDPID(), 1+portsInBind.get(binding.getSwitchPort().getSwitchDPID()));
		}
		mb.setExact(MatchField.ETH_SRC, binding.getMacAddress());
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		mb.setExact(MatchField.IPV6_SRC, (IPv6Address)binding.getAddress());
		mb.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
		List<OFInstruction> instructions = new ArrayList<>();
		instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
		doFlowAdd(binding.getSwitchPort().getSwitchDPID(), STATIC_TABLE_ID, mb.build(), null, instructions, BINDING_LAYER_PRIORITY);
		rank.put(binding.getSwitchPort(), BINDING_LAYER_PRIORITY);
		hostWithPort.put(binding.getSwitchPort(), (int)(binding.getMacAddress().getLong()));
	}
	protected void doUnbindIPv4(UnbindIPv4Action action) {
		manager.delBinding(action.getIpv4Address());
		Binding<?> binding = action.getBinding();
		if(securityPort.contains(binding.getSwitchPort())){
			return;
		}
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		mb.setExact(MatchField.ETH_SRC, binding.getMacAddress());
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv4);
		mb.setExact(MatchField.IPV4_SRC, (IPv4Address)binding.getAddress());
		mb.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
		doFlowRemove(binding.getSwitchPort().getSwitchDPID(), STATIC_TABLE_ID, mb.build());
	}
	protected void doUnbindIPv6(UnbindIPv6Action action) {
		manager.delBinding(action.getIPv6Address());
		System.out.println("发生解绑");
		Binding<?> binding = action.getBinding();
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		mb.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
		doFlowRemove(binding.getSwitchPort().getSwitchDPID(), STATIC_TABLE_ID, mb.build());
	}
	protected boolean doCheckIPv4BInding(CheckIPv4BindingAction action){
		return manager.check(action.getSwitchPort(), action.getMacAddress(), action.getIPv4Address());
	}
	protected boolean doCheckIPv6Binding(CheckIPv6BindingAction action) {
		SwitchPort switchPort=manager.getSwitchPort(action.getIPv6Address());
		MacAddress macAddress=manager.getMacAddress(action.getIPv6Address());
		return manager.check(action.getSwitchPort(), action.getMacAddress(), action.getIPv6Address());
	}
	protected void doMeterMod(DatapathId switchId, long meterId,List<OFMeterBand> bands) {
		OFMeterMod.Builder meterBuilder = OFFactories.getFactory(OFVersion.OF_14).buildMeterMod()
				.setMeterId(meterId)
				.setBands(bands)
				.setCommand(0);
		IOFSwitch sw = switchService.getSwitch(switchId);
		if(sw!= null){
			sw.write(meterBuilder.build());
		}
	}
	protected void doFlowMod(FlowModAction action) {
		doFlowMod(action.getSwitchId(),
				action.getTableId(),
				action.getMatch(),
				action.getActions(),
				action.getInstructions(),
				action.getPriority(), 
				action.getHardTimeout(), action.getIdleTimeout());
	}
	protected void doFlowMod(DatapathId switchId,TableId tableId,Match match, List<OFAction> actions, List<OFInstruction> instructions,int priority){
		OFFlowModify.Builder fab = OFFactories.getFactory(OFVersion.OF_14).buildFlowModify();
		fab.setCookie(cookie)
		   .setTableId(tableId)
		   .setHardTimeout(0)
		   .setIdleTimeout(0)
		   .setPriority(priority)
		   .setBufferId(OFBufferId.NO_BUFFER)
		   .setMatch(match);
		if(instructions == null){
			instructions=new ArrayList<>();
		}
		if(actions != null){
			OFInstructionApplyActions output = OFFactories.getFactory(OFVersion.OF_14).instructions()
					.buildApplyActions().setActions(actions).build();
			instructions.add(output);
		}
		fab.setInstructions(instructions);
		IOFSwitch sw = switchService.getSwitch(switchId);
		if(sw!= null){
			sw.write(fab.build());
		}
	}
	protected void doFlowMod(DatapathId switchId,TableId tableId,Match match, List<OFAction> actions, List<OFInstruction> instructions,int priority, int hardTimeout,int idleTimeout){
		OFFlowModifyStrict.Builder fab = OFFactories.getFactory(OFVersion.OF_14).buildFlowModifyStrict();
		fab.setCookie(cookie)
		.setTableId(tableId)
		.setHardTimeout(hardTimeout)
		.setIdleTimeout(idleTimeout)
		.setPriority(priority)
		.setBufferId(OFBufferId.NO_BUFFER)
		.setMatch(match);
		if(instructions == null){
			instructions=new ArrayList<>();
		}
		if(actions != null){
			OFInstructionApplyActions output = OFFactories.getFactory(OFVersion.OF_14).instructions()
					.buildApplyActions().setActions(actions).build();
			instructions.add(output);
		}
		fab.setInstructions(instructions);
		IOFSwitch sw = switchService.getSwitch(switchId);
		if(sw!= null){
			sw.write(fab.build());
		}
	}
	protected void doFlowAdd(FlowAddAction action) {
		doFlowAdd(action.getSwitchId(), 
				action.getTableId(), 
				action.getMatch(),
				action.getActions(), 
				action.getInstructions(),
				action.getPriority());
	}
	@Override
	public void doFlowAdd(DatapathId switchId,TableId tableId,Match match, List<OFAction> actions, List<OFInstruction> instructions,int priority) {
		doFlowAdd(switchId, tableId, match, actions, instructions, priority,0,0);
	}
	protected void doFlowAdd(DatapathId switchId,TableId tableId,Match match, List<OFAction> actions, List<OFInstruction> instructions,int priority,int hardTimeout, int idleTimeout){
		OFFlowAdd.Builder fab = OFFactories.getFactory(OFVersion.OF_14).buildFlowAdd();
		fab.setCookie(cookie)
		   .setTableId(tableId)
		   .setHardTimeout(hardTimeout)
		   .setIdleTimeout(idleTimeout)
		   .setPriority(priority)
		   .setBufferId(OFBufferId.NO_BUFFER)
		   .setMatch(match);
		if(instructions == null){
			instructions=new ArrayList<>();
		}
		if(actions != null){
			OFInstructionApplyActions output = OFFactories.getFactory(OFVersion.OF_14).instructions()
					.buildApplyActions().setActions(actions).build();
			instructions.add(output);
		}
		fab.setInstructions(instructions);
		IOFSwitch sw = switchService.getSwitch(switchId);
		if(sw!= null){
			sw.write(fab.build());
		}
	}
	protected void doFlowRemove(FlowRemoveAction action) {
		doFlowRemove(action.getSwitchId(),
				action.getTableId(),
				action.getMatch());
	}
	@Override
	public void doFlowRemove(DatapathId switchId, TableId tableId, Match match) {
		OFFlowDelete.Builder fdb = OFFactories.getFactory(OFVersion.OF_14).buildFlowDelete();
		fdb.setMatch(match)
		   .setCookie(cookie)
		   .setTableId(tableId)
		   .setBufferId(OFBufferId.NO_BUFFER);
		IOFSwitch sw = switchService.getSwitch(switchId);
		if(sw!= null){
			sw.write(fdb.build());
		}
	}
	public void doFlowRemove(DatapathId switchId, TableId tableId, Match match,int priority) {
		OFFlowDeleteStrict.Builder fdb = OFFactories.getFactory(OFVersion.OF_14).buildFlowDeleteStrict();
		fdb.setMatch(match)
		.setCookie(cookie)
		.setTableId(tableId)
		.setPriority(priority)
		.setBufferId(OFBufferId.NO_BUFFER);
		IOFSwitch sw = switchService.getSwitch(switchId);
		if(sw!= null){
			sw.write(fdb.build());
		}
	}
	@Override
	public boolean addSecurityPort(SwitchPort switchPort) {
		IOFSwitch sw = switchService.getActiveSwitch(switchPort.getSwitchDPID());
		if(sw!=null){
			Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
			mb.setExact(MatchField.IN_PORT, switchPort.getPort());
			List<OFInstruction> instructions = new ArrayList<>();
			instructions.add(OFFactories.getFactory(OFVersion.OF_14).instructions().gotoTable(FLOW_TABLE_ID));
			doFlowMod(switchPort.getSwitchDPID(), STATIC_TABLE_ID, mb.build(), null, instructions, BINDING_LAYER_PRIORITY);
		}
		return securityPort.add(switchPort);
	}
	@Override
	public boolean delSecurityPort(SwitchPort switchPort) {
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		mb.setExact(MatchField.IN_PORT, switchPort.getPort());
		doFlowRemove(switchPort.getSwitchDPID(), STATIC_TABLE_ID, mb.build());
		return securityPort.remove(switchPort);
	}
	@Override
	public Set<SwitchPort> getSecurityPorts() {
		return securityPort;
	}
	@Override
	public List<Binding<?>> getBindings() {
		return manager.getBindings();
	}
	@Override
	public void topologyChanged(List<LDUpdate> linkUpdates) {
		updateQueue.addAll(linkUpdates);
	}
	@Override
	public void linkDiscoveryUpdate(LDUpdate update) {
		updateQueue.add(update);
	}
	@Override
	public void linkDiscoveryUpdate(List<LDUpdate> updateList) {
		updateQueue.addAll(updateList);
	}
	@Override
	public Map<DatapathId, Integer> getPortsInBind(){
		return portsInBind;
	}
	@Override
	public Map<SwitchPort, Integer> getRank(){
		return rank;
	}
	@Override
	public Set<DatapathId> getStaticSwId() {
		return staticSwId;
	}
	@Override
	public Map<SwitchPort, Integer> getHostWithPort(){
		return hostWithPort;
	}
	@Override
	public Queue<SwitchPort> getNormalPorts() {
		return normalPorts;
	}
	@Override
	public Queue<SwitchPort> getAbnormalPorts() {
		return abnormalPorts;
	}
	@Override
	public Map<SwitchPort, Integer> getObservePorts() {
		return observePorts;
	}
	@Override
	public Set<SwitchPort> getPushFlowToSwitchPorts() {
		return pushFlowToSwitchPorts;
	}
}
