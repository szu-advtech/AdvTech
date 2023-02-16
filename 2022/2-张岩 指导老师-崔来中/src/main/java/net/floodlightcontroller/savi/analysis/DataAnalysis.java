package net.floodlightcontroller.savi.analysis;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.Thread.State;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFlowStatsEntry;
import org.projectfloodlight.openflow.protocol.OFFlowStatsReply;
import org.projectfloodlight.openflow.protocol.OFPortStatsEntry;
import org.projectfloodlight.openflow.protocol.OFPortStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsRequest;
import org.projectfloodlight.openflow.protocol.OFStatsType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.protocol.ver13.OFMeterSerializerVer13;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.OFGroup;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.primitives.UnsignedLong;
import com.google.common.util.concurrent.ListenableFuture;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.savi.analysis.web.AnalysisWebRoutable;
import net.floodlightcontroller.savi.binding.Binding;
import net.floodlightcontroller.savi.flow.FlowAction;
import net.floodlightcontroller.savi.flow.FlowAction.FlowActionFactory;
import net.floodlightcontroller.savi.service.SAVIProviderService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
public class DataAnalysis implements IFloodlightModule, IAnalysisService {
	private static final Logger log=LoggerFactory.getLogger(DataAnalysis.class);
	private static IThreadPoolService threadPoolService;
	private static IOFSwitchService switchService;
	private static IRestApiService restApiService;
	private static SAVIProviderService saviProvider;
	private static ScheduledFuture<?> flowSchedule;
	private static ScheduledFuture<?> portPackets;
	private static int flowStasInterval = 1;
	public static boolean isEnable= false;
	private static final String ENABLED_STR = "enable";
	public static boolean isSecurity = true;
	private static final String SECURE_STR = "secure";
	public static int STATUS = 1;
	public static final int INIT_STAGE = 0;
	public static final int PLAN_LOSSRATE = 1;
	public static final int PLAN_TRAFFIC = 2;
	public static final int RELIABLE_PORT_PRIORITY = 200;
	static final int BINDING_PRIORITY = 5;
	private static String filePath = "E:/maxtraffic.txt";
	private static String filePath2 = "E:/savilog/";
	private static String filePath5 ="E:/savilog/abnormalLog.txt";
	private static final Map<SwitchPort, U64> inPortPackets = new ConcurrentHashMap<SwitchPort, U64>();
	private static final Map<SwitchPort, U64> inTentativePortPackets = new ConcurrentHashMap<SwitchPort, U64>();
	private static final Map<SwitchPort, U64> inPortPacketsRes = new ConcurrentHashMap<SwitchPort, U64>();
	private static final Map<SwitchPort, U64> outPortPackets = new ConcurrentHashMap<SwitchPort, U64>();
	private static final Map<SwitchPort, U64> outTentativePortPackets = new ConcurrentHashMap<SwitchPort, U64>();
	private static final Map<SwitchPort, U64> outPortPacketsRes = new ConcurrentHashMap<SwitchPort, U64>();
	private static final Map<SwitchPort , PacketOfFlow> packetOfFlows = new ConcurrentHashMap<>();
	private static final Map<Integer, Double> maxTraffics = new ConcurrentHashMap<>();
	private static final Map<Integer, Double> outTraffics = new ConcurrentHashMap<>();
	private SingletonTask initTimer;
	private double LOSS_RATE_THRESHOLD = 0.2;
	private int LOSS_NUM_THRESHOLD = 100;
	private Set<SwitchPort> pickFromNormal = new HashSet<>();
	private Set<SwitchPort> rightPorts = new HashSet<>();
	@SuppressWarnings("all")
	private static ScheduledFuture<?> normalPortSchedule;
	@SuppressWarnings("all")
	private static ScheduledFuture<?> abnormalPortSchedule;
	@SuppressWarnings("all")
	private static ScheduledFuture<?> observePortSchedule;
	private SingletonTask testTimer;
	private SingletonTask flowlogTimer;
	private SingletonTask checkRules;
	private static TableId STATIC_TABLE_ID=TableId.of(0);
	private static TableId DYNAMIC_TABLE_ID=TableId.of(1);
	private static TableId FLOW_TABLE_ID = TableId.of(2);
	private Map<SwitchPort, Integer> rank;
	private Map<DatapathId, Integer> portsInBind;
	private Map<DatapathId, Boolean> convertFlag		=new HashMap<>();
	private short cycleTime=5;
	private short period=20;
	private Map<SwitchPort, U64> packetsInPeriod	= new HashMap<>();
	private Map<SwitchPort, U64> packetsInRecord	=new HashMap<>();
	private boolean timeToSave=true;
	private Set<SwitchPort> portsList				=new HashSet<>();
	private int priorityLevel=96;
	private Set<DatapathId> staticSwId;
	private long stableTime	;
	private Map<SwitchPort, Integer> hostsCredit		=new HashMap<>();
	private Map<SwitchPort, Boolean> logFlag=		new HashMap<>();
	private boolean alreadyInit =false;
	private Map<DatapathId, Integer> dynamicRuleNumber=new HashMap<>();
	private Map<DatapathId, Integer> staticRuleNumber=new HashMap<>();
	private Map<DatapathId, Integer> ruleNum =new HashMap<>();
	private boolean autoCheck=false;
	private SimpleDateFormat sdflog = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> services=new ArrayList<Class<? extends IFloodlightService>>();
		services.add(IAnalysisService.class);
		return services;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		Map<Class<? extends IFloodlightService>, IFloodlightService> map=new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
		map.put(IAnalysisService.class, this);
		return map;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l=new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IThreadPoolService.class);
		l.add(IOFSwitchService.class);
		l.add(IRestApiService.class);
		l.add(SAVIProviderService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		switchService=context.getServiceImpl(IOFSwitchService.class);
		threadPoolService=context.getServiceImpl(IThreadPoolService.class);
		restApiService=context.getServiceImpl(IRestApiService.class);
		saviProvider=context.getServiceImpl(SAVIProviderService.class);
		Map<String, String> config = context.getConfigParams(this);
		if(config.containsKey(ENABLED_STR)){
			try {
				isEnable = Boolean.parseBoolean(config.get(ENABLED_STR).trim());
			} catch (Exception e) {
				log.error("Could not parse '{}'. Using default of {}", ENABLED_STR, isEnable);
			}
		}
		if(config.containsKey(SECURE_STR)){
			try {
				isSecurity = Boolean.parseBoolean(config.get(SECURE_STR).trim());
			} catch (Exception e) {
				log.error("Could not parse '{}'. Using default of {}", SECURE_STR, isSecurity);
			}
		}
	}
	@Override
	public void startUp(FloodlightModuleContext context) throws FloodlightModuleException {
		restApiService.addRestletRoutable(new AnalysisWebRoutable());
		ScheduledExecutorService ses1 = threadPoolService.getScheduledExecutor();
		testTimer = new SingletonTask(ses1, new Runnable() {
			@Override
			public void run() {
				if(alreadyInit) {
					StringBuffer sb = new StringBuffer();
					testPortSet(sb, normalPorts, "normal");
					testPortSet(sb, observePorts.keySet(), "observe");
					testPortSet(sb, abnormalPorts, "abnormal");
					log.info(sb.toString());
				}
				testTimer.reschedule(500, TimeUnit.MILLISECONDS);
			}
		});
		testTimer.reschedule(3, TimeUnit.SECONDS);
		ScheduledExecutorService ses3 = threadPoolService.getScheduledExecutor();
		flowlogTimer = new SingletonTask(ses3, new Runnable() {
			@Override
			public void run() {
				if(alreadyInit) {
					StringBuffer sb = new StringBuffer();
					double[] tempV = new double[rank.size() + 1];
					for(SwitchPort swport : rank.keySet()){
						if(swport.getPort().getPortNumber()>0){
							U64 u=inPortPacketsRes.get(swport)==null?U64.ZERO:inPortPacketsRes.get(swport);
							int terminatorNum = computeTerminatorNum(swport);
							if(terminatorNum==-1) continue;
							tempV[terminatorNum] = v;
						}
					}
					String[] splits =  sdflog.format(System.currentTimeMillis()).split(" ");
					sb.append(splits[1] + " ");
					for(int i = 1; i < tempV.length; i++){
						sb.append(tempV[i]);
						if(i != tempV.length-1){
							sb.append(",");
						}
					}
					sb.append("\r\n");
					writeToTxt(filePath2 + splits[0] +".txt", true, sb.toString());
				}
			}
		});
		flowlogTimer.reschedule(5, TimeUnit.SECONDS);
		ScheduledExecutorService ses4 = threadPoolService.getScheduledExecutor();
		checkRules= new SingletonTask(ses4, new Runnable() {
			@Override
			public void run() {
				if(autoCheck) {
					for(DatapathId dpid : staticSwId) 
						convertTable(dpid, false);
				}
				checkRules.reschedule(20, TimeUnit.SECONDS);
			}
		});
		checkRules.reschedule(60, TimeUnit.SECONDS);
	}
	@Override
	public void updateOutFlow() {
		for(Map.Entry<SwitchPort, U64> entry : outPortPacketsRes.entrySet()){
			SwitchPort sp = entry.getKey();
			if(rank.containsKey(sp)){
				outTraffics.put(computeTerminatorNum(sp),curOut);
			}
		}
		StringBuffer sb = new StringBuffer();
		DecimalFormat df = new DecimalFormat("####0.00");
		for(Map.Entry<Integer, Double> entry : outTraffics.entrySet()){
			sb.append(entry.getKey() + " " + df.format(entry.getValue()) + "\r\n");
		}
		log.info("当前出口流量集合如下：\r\n" + sb.toString() + "====================");
	}
	public Object showOutFlow() {
		Map<Integer, Double> map = new HashMap<>();
		for(Map.Entry<SwitchPort, U64> entry : outPortPacketsRes.entrySet()){
			SwitchPort sp = entry.getKey();
			if(sp.getSwitchDPID().toString().contains("1")) continue;
			if(rank.containsKey(sp)){
			}
		}
		StringBuffer sb = new StringBuffer();
		DecimalFormat df = new DecimalFormat("####0.00");
		for(Map.Entry<Integer, Double> entry : map.entrySet()){
			sb.append(entry.getKey() + " " + df.format(entry.getValue()) + "\r\n");
		}
		log.info("当前出口流量集合如下：\r\n" + sb.toString() + "====================");
		return map;
	}
	private void trafficToMap(){
		for(int i = 1; i <= rank.size(); i++){
			maxTraffics.put(i, 100.0);
		}
	}
	private void trafficToMap(String filename){
		try {
			synchronized (filename) {
				File file = new File(filename);
				if(!file.exists()){
					log.info("文件不存在");
					file.createNewFile();
					trafficToMap();
					return;
				}
				BufferedReader in = new BufferedReader(new FileReader(filename));
				try {
					String temp = null;
					boolean flag = false;
					while((temp = in.readLine()) != null){
						flag = true;
						String[] splits = temp.split(" ");
						if(splits.length != 2 || !isInteger(splits[0]) || !isDouble(splits[1])) {
							trafficToMap();
							return;
						}
						else {
							maxTraffics.put(Integer.parseInt(splits[0]), Double.parseDouble(splits[1]));
						}
					}
					if(!flag){
						trafficToMap();
					}
					else {
						for( Map.Entry<Integer, Double> entry : maxTraffics.entrySet()){
							System.out.println(entry.getKey() + " " + entry.getValue());
						}
					}
					in.close();
				} catch (Exception ex){
					ex.printStackTrace();
				}
				finally {
					if(in != null){
						try {
							in.close();
						}catch(IOException e){}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	private boolean isDouble(String str) {
		if (null == str || "".equals(str)) {  
	        return false;  
	    }  
	    return pattern.matcher(str).matches();  
	}
	private boolean isInteger(String str) {
		if (null == str || "".equals(str)) {  
	        return false;  
	    }  
	    return pattern.matcher(str).matches(); 
	}
	private void testPortSet(StringBuffer sb , Collection<SwitchPort> switchPorts, String name){
		if(sb == null) return ;
		sb.append(name + "[ ");
		for(SwitchPort sp : switchPorts){
			if(rank.containsKey(sp))
				sb.append(computeTerminatorNum(sp) + ",");
		}
		sb.deleteCharAt(sb.length()-1);
		sb.append(" ]" + "  ");
	}
	private void firstStage(){
		trafficToMap(filePath);
		ScheduledExecutorService ses0=threadPoolService.getScheduledExecutor();
			@Override
			public void run(){
				List<SwitchPort> tmp=new ArrayList<>(rank.keySet());
				Collections.shuffle(tmp);
				normalPorts.addAll(tmp);
				for(DatapathId switchId : portsInBind.keySet()) {
					IOFSwitch sw=switchService.getSwitch(switchId);
					Collection<OFPort> ports=sw.getEnabledPortNumbers();
					staticRuleNumber.put(switchId, 5+ports.size());
					for(OFPort port : ports) {
						if(rank.containsKey(new SwitchPort(switchId, port))) continue;
						Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
						mb.setExact(MatchField.IN_PORT, port);
						List<OFInstruction> instructions = new ArrayList<>();
						instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(FLOW_TABLE_ID));
						log.info("add reliable switch port static match rule"+switchId+"----"+port);
						saviProvider.doFlowAdd(switchId, STATIC_TABLE_ID, mb.build(), null, instructions, RELIABLE_PORT_PRIORITY);
					}
				}
				for(SwitchPort switchPort : rank.keySet()) {
					hostsCredit.put(switchPort, 24);
					logFlag.put(switchPort, true);
				}
				enableAnalysis(true);
				StringBuffer sb = new StringBuffer();
				testPortSet(sb, normalPorts, "normal");
				testPortSet(sb, observePorts.keySet(), "observe");
				testPortSet(sb, abnormalPorts, "abnormal");
				log.info("第一次得到的主机集合"+sb.toString());
				enablePortHandle();
				stableTime=System.currentTimeMillis();
				alreadyInit=true;
			}
		});
		initTimer.reschedule(8, TimeUnit.SECONDS);
	}
	private void doFlowRemove(Set<SwitchPort> switchPorts) {
		List<FlowAction> removeActions = new ArrayList<>();
		for(SwitchPort switchPort : switchPorts) {
			Match.Builder mb=OFFactories.getFactory(OFVersion.OF_13).buildMatch();
			mb.setExact(MatchField.IN_PORT, switchPort.getPort());
			removeActions.add(FlowActionFactory.getFlowRemoveAction(switchPort.getSwitchDPID(), DYNAMIC_TABLE_ID, mb.build()));
			packetOfFlows.get(switchPort).init();
			dynamicRuleNumber.remove(switchPort.getSwitchDPID(),dynamicRuleNumber.get(switchPort.getSwitchDPID())-2);
		}
		saviProvider.pushFlowActions(removeActions);
	}
	private void doFlowAdd(Set<SwitchPort> switchPorts){
		List<Binding<?>> bindings = saviProvider.getBindings();
		List<FlowAction> addActions = new ArrayList<>();
		for(Binding<?> binding : bindings) {
			if(switchPorts.contains(binding.getSwitchPort())){
				Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
				Match.Builder mb2 = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
				mb.setExact(MatchField.ETH_SRC, binding.getMacAddress());
				mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
				mb.setExact(MatchField.IPV6_SRC, (IPv6Address)binding.getAddress());
				mb.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
				mb2.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
				List<OFInstruction> instructions = new ArrayList<>();
				instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(FLOW_TABLE_ID));
				addActions.add(FlowActionFactory.getFlowAddAction(
						binding.getSwitchPort().getSwitchDPID(), 
						DYNAMIC_TABLE_ID, mb.build(), null, instructions, rank.get(binding.getSwitchPort())));
				addActions.add(FlowActionFactory.getFlowAddAction(
						binding.getSwitchPort().getSwitchDPID(), 
						DYNAMIC_TABLE_ID, mb2.build(), null, null, rank.get(binding.getSwitchPort())-1));
				switchPorts.remove(binding.getSwitchPort());
				if(!dynamicRuleNumber.containsKey(binding.getSwitchPort().getSwitchDPID())) dynamicRuleNumber.put(binding.getSwitchPort().getSwitchDPID(), 0);
				dynamicRuleNumber.put(binding.getSwitchPort().getSwitchDPID(), 2+dynamicRuleNumber.get(binding.getSwitchPort().getSwitchDPID()));
			}
		}
		if(switchPorts.size() != 0){
			log.error("DA doFlowAdd line:509 ，switchPorts没有处理完，还剩{}",switchPorts);
		}
		saviProvider.pushFlowActions(addActions);
	}
	private void doFlowMod(Set<SwitchPort> switchPorts){
		doFlowMod(switchPorts, DYNAMIC_TABLE_ID);
	}
	private void doFlowMod(Set<SwitchPort> switchPorts, TableId tableId){
		List<Binding<?>> bindings = saviProvider.getBindings();
		List<FlowAction> modActions = new ArrayList<>();
		for(Binding<?> binding : bindings) {
			if(switchPorts.contains(binding.getSwitchPort())){
				Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
				Match.Builder mb2 = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
				mb.setExact(MatchField.ETH_SRC, binding.getMacAddress());
				mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
				mb.setExact(MatchField.IPV6_SRC, (IPv6Address)binding.getAddress());
				mb.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
				List<OFInstruction> instructions = new ArrayList<>();
				instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(FLOW_TABLE_ID));
				if(tableId.equals(DYNAMIC_TABLE_ID)) {
					mb2.setExact(MatchField.IN_PORT, binding.getSwitchPort().getPort());
					modActions.add(FlowActionFactory.getFlowModAction(binding.getSwitchPort().getSwitchDPID(), 
							tableId, mb2.build(), null, null, rank.get(binding.getSwitchPort())-1,0,0));
				}
				modActions.add(FlowActionFactory.getFlowModAction(binding.getSwitchPort().getSwitchDPID(), 
						tableId, mb.build(), null, instructions, rank.get(binding.getSwitchPort()), 0, 0));
				switchPorts.remove(binding.getSwitchPort());
			}
		}
		if(switchPorts.size() != 0){
			log.error("DA doFlowAdd line:509 ，switchPorts没有处理完，还剩{}",switchPorts);
		}
		saviProvider.pushFlowActions(modActions);
	}
	private void startStatisticsCollector() {
		flowSchedule=threadPoolService.getScheduledExecutor().scheduleAtFixedRate(
		portPackets = threadPoolService.getScheduledExecutor().scheduleAtFixedRate(
	}
	@Override
	public synchronized void changeStatusByRest(int flag) {
		if(flag == INIT_STAGE) {
			System.out.println("边缘交换机端口：" + rank.keySet().size());
			for(DatapathId dpid : portsInBind.keySet()) {
				System.out.println(dpid.getLong());
			}
			firstStage();
		}
	}
	@Override
	public synchronized void enableAnalysis(boolean flag) {
		if (flag && !isEnable) {
			startStatisticsCollector();
			isEnable = true;
		} else if (!flag && isEnable) {
			stopStatsCollector();
			isEnable = false;
		}
	}
	private void stopStatsCollector(){
		if(! flowSchedule.cancel(false)) {
			log.error("Could not cancel flow stats thread");
		}
		else {
			log.warn("Statistics collection thread(s) stopped");
		}
		if(! portPackets.cancel(false)) {
			log.error("Could not cancel flow stats thread");
		}
		else {
			log.warn("Statistics collection thread(s) stopped");
		}
	}
	public void enablePortHandle() {
		normalPortSchedule = threadPoolService.getScheduledExecutor().
		observePortSchedule =threadPoolService.getScheduledExecutor().
		abnormalPortSchedule = threadPoolService.getScheduledExecutor()
	}
	private class NormalPortThread extends Thread {
		@Override
		public void run() {
			SwitchPort cur = null;
			Set<SwitchPort> handleSet = new HashSet<>();
			int t1=0;
			int t2=0;
			for(int i = 0 ; i < 4-staticSwId.size() ; i++){
				cur = normalPorts.poll();
				if(cur == null) break;
				if(saviProvider.getPushFlowToSwitchPorts().contains(cur)) {
					normalPorts.offer(cur);
					++t1;
					if(normalPorts.size()==t1) break;
					i--;
					continue;
				}
				if(staticSwId.contains(cur.getSwitchDPID())) {
					t2++;
					normalPorts.offer(cur);
					if(normalPorts.size()==t2) break;
					i--;
					continue;
				}
				handleSet.add(cur);
				observePorts.put(cur, 0);
				pickFromNormal.add(cur);
			}
			doFlowAdd(handleSet);
		}
	}
	private class AbnormalPortThread extends Thread {
		@Override
		public void run() {
			int curSize = abnormalPorts.size();
			for(int i = 0 ; i < curSize ; i++){
				SwitchPort cur = abnormalPorts.poll();
				if(cur == null) break;
				if(rightPorts.contains(cur)){
					observePorts.put(cur, 0);
					convert(cur.getSwitchDPID());
				}                                                           
				else {
					abnormalPorts.offer(cur);
				}
			}
		}
	}
	private class ObservePortThread extends Thread {
		@Override
		public void run() {
			Set<SwitchPort> actionPorts = new HashSet<>();
			Iterator<Map.Entry<SwitchPort, Integer>> iterator = observePorts.entrySet().iterator();
			while(iterator.hasNext()){
				Map.Entry<SwitchPort, Integer> entry = iterator.next();
				if(rightPorts.contains(entry.getKey())){
					if(entry.getValue() >= 6-hostsCredit.get(entry.getKey())/8) {
						normalPorts.offer(entry.getKey());
						if(pickFromNormal.contains(entry.getKey()))
							pickFromNormal.remove(entry.getKey());
						if(!saviProvider.getPushFlowToSwitchPorts().contains(entry.getKey()))
							actionPorts.add(entry.getKey());
						iterator.remove();
					}
					else {
						observePorts.put(entry.getKey(), entry.getValue() + 1);
					}
				}
				else {
					abnormalPorts.offer(entry.getKey());
					convert(entry.getKey().getSwitchDPID());
					if(pickFromNormal.contains(entry.getKey()))
						pickFromNormal.remove(entry.getKey());
					iterator.remove();
				}
			}
			doFlowRemove(actionPorts);
		}
	}
	private boolean isNormal(SwitchPort switchPort) {
		if(packetOfFlows.get(switchPort)==null) return true;
		if(packetOfFlows.get(switchPort).getDropRate()!=1&&packetOfFlows.get(switchPort).getDropRate() > LOSS_RATE_THRESHOLD) {
			int t=hostsCredit.get(switchPort)-2>0?hostsCredit.get(switchPort)-2:0;
			hostsCredit.put(switchPort, t);
			if(logFlag.get(switchPort)) {
				writeErrorLog(switchPort, true);
				logFlag.put(switchPort, false);
			}
			return false;
		}
		if(packetOfFlows.get(switchPort).getDropNum() > LOSS_NUM_THRESHOLD) {
			int t=hostsCredit.get(switchPort)-2>0?hostsCredit.get(switchPort)-2:0;
			hostsCredit.put(switchPort, t);
			if(logFlag.get(switchPort)) {
				writeErrorLog(switchPort, true);
				logFlag.put(switchPort, false);
			}
			return false;
		}
		int t=hostsCredit.get(switchPort)+1<47?hostsCredit.get(switchPort)+1:47;
		hostsCredit.put(switchPort, t);
		if(!logFlag.get(switchPort)) {
			writeErrorLog(switchPort, false);
			logFlag.put(switchPort, true);
		}
		return true;
	}
	private Set<SwitchPort> getChangeNormalPorts(int type){
		rightPorts.clear();
		if(type == PLAN_LOSSRATE){
			for(SwitchPort switchPort : rank.keySet()){
				if(isNormal(switchPort)) {
					rightPorts.add(switchPort);
				}
			}
			return rightPorts;
		}
		else if(type == PLAN_TRAFFIC){
			Map<Integer, Double> testMap = new HashMap<>();
			boolean flag = false;
			for(Map.Entry<SwitchPort, U64> entry : inPortPacketsRes.entrySet()){
				SwitchPort swport = entry.getKey();
				if(rank.containsKey(swport)){
					int terminatorNum = computeTerminatorNum(swport);
					testMap.put(terminatorNum, maxTraffics.get(terminatorNum));
					if(v > maxTraffics.get(terminatorNum)){
						flag = true;
						log.info("当前入流量" + v + "超过历史峰值" + maxTraffics.get(terminatorNum));
							rightPorts.add(swport);
							log.info("正常大流量 curOut：" + curOut + " map：" + outTraffics.get(terminatorNum));
							maxTraffics.put(terminatorNum, v);
						}
						else{
							flag = false;
						}
					}
					else {
						rightPorts.add(swport);
					}
				}
			}
			if(flag){
				StringBuffer sb = new StringBuffer();
				for(Map.Entry<Integer, Double> entry : maxTraffics.entrySet()){
					sb.append(entry.getKey() + " " + entry.getValue() + "\r\n");
				}
				System.out.println("=============更新峰值文件，h1：" + maxTraffics.get(1) + "=============");
				writeToTxt(filePath, false, sb.toString());
			}
			return rightPorts;
		}
		return null;
	}
	private int getOutPacketsByPort(SwitchPort sp){
		if(outTraffics.size() < 1){
			for(int i = 1; i <= rank.size(); i++){
				outTraffics.put(i, 0.0);
			}
		}
		long pnum = 0;
		for(SwitchPort cur : outPortPacketsRes.keySet()){
			if(cur.getSwitchDPID().equals(sp.getSwitchDPID()) && rank.containsKey(cur)){
				pnum += outPortPacketsRes.get(cur).getValue();
			}
		}
		return (int) pnum;
	}
	private class FlowStatisCollector extends Thread{
		@Override
		public void run(){
			Map<DatapathId, List<OFStatsReply>> map = getSwitchStatistics(portsInBind.keySet(), OFStatsType.FLOW);
			for(Map.Entry<DatapathId, List<OFStatsReply>> entry : map.entrySet()){
				DatapathId swid = entry.getKey();
				if(swid == null || entry.getValue() == null) continue;
				for(OFStatsReply r :entry.getValue()){
					OFFlowStatsReply psr = (OFFlowStatsReply) r;
					for(OFFlowStatsEntry psrEntry : psr.getEntries()){
						int priority = psrEntry.getPriority();
						OFPort port = psrEntry.getMatch().get(MatchField.IN_PORT);
						if(port == null) continue;
						SwitchPort swport = new SwitchPort(swid , port);
						PacketOfFlow packetOfFlow = packetOfFlows.get(swport);
						if(psrEntry.getMatch().isExact(MatchField.IPV6_SRC)){
							if(packetOfFlow == null){
								packetOfFlow= new PacketOfFlow(psrEntry.getPacketCount().getValue(), 0 , psrEntry.getPacketCount().getValue() , 0 , 0 , 0);
							}else{
								packetOfFlow.setPassNum(psrEntry.getPacketCount().getValue() - packetOfFlow.getAccumulatePassNum());
								packetOfFlow.setAccumulatePassNum(psrEntry.getPacketCount().getValue());
								long fenmu1 = packetOfFlow.getDropNum() + packetOfFlow.getPassNum();
								long fenmu2 = packetOfFlow.getAccumulateDropNum() + packetOfFlow.getAccumulatePassNum();
							}
						}else{
							if(packetOfFlow == null){
								packetOfFlow=new PacketOfFlow(0, psrEntry.getPacketCount().getValue(),0, psrEntry.getPacketCount().getValue() , 0 , 0);
							}else{
								packetOfFlow.setDropNum(psrEntry.getPacketCount().getValue()-packetOfFlow.getAccumulateDropNum());
								packetOfFlow.setAccumulateDropNum(psrEntry.getPacketCount().getValue());
								long fenmu1 = packetOfFlow.getDropNum() + packetOfFlow.getPassNum();
								long fenmu2 = packetOfFlow.getAccumulateDropNum() + packetOfFlow.getAccumulatePassNum();
							}
						}
						packetOfFlows.put(swport, packetOfFlow);
					}
				}
			}
			getChangeNormalPorts(STATUS);
		}
	}
	private void classify(SwitchPort switchPort) {
		if(observePorts.containsKey(switchPort)) {
			if(observePorts.get(switchPort)>=2) {
				normalPorts.offer(switchPort);
				observePorts.remove(switchPort);
				Match.Builder mb=OFFactories.getFactory(OFVersion.OF_13).buildMatch();
				mb.setExact(MatchField.IN_PORT, switchPort.getPort());
				saviProvider.doFlowRemove(switchPort.getSwitchDPID(), DYNAMIC_TABLE_ID, mb.build());
			}else {
				observePorts.put(switchPort, observePorts.get(switchPort)+1);
			}
		}else if(handleSet.contains(switchPort)) {
			normalPorts.offer(switchPort);
		}else {
			abnormalPorts.remove(switchPort);
			observePorts.put(switchPort, 0);
		}
	}
	private void writeToTxt(String filePath, boolean append, String text) {	
        RandomAccessFile fout = null;
        FileChannel fcout = null;
        try {
        	File file = new File(filePath);
        	if(!file.exists()){
        		file.createNewFile();
        	}
            fout = new RandomAccessFile(file, "rw");
            if(append){
            	fout.seek(filelength);
            }
            else{
            	fout.seek(0);
            }
            FileLock flout = null;
            try {
                flout = fcout.tryLock();
            } catch (Exception e) {
                System.out.print("lock is exist ......");
                return;
            }
            flout.release();
            fcout.close();
            fout.close();
        } catch (IOException e1) {
            e1.printStackTrace();
            System.out.print("file no find ...");
        } finally {
            if (fcout != null) {
                try {
                    fcout.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    fcout = null;
                }
            }
            if (fout != null) {
                try {
                    fout.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    fout = null;
                }
            }
        }
	}
		try {
			synchronized (filePath) {
				BufferedWriter out = new BufferedWriter(new FileWriter(filePath, append));
				try {
					out.write(text);
					out.flush();
				} finally {
					out.close();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	private class GetStatisticsThread extends Thread{
		List<OFStatsReply> replies;
		DatapathId dpId;
		OFStatsType statsType;
		public GetStatisticsThread(DatapathId dpId , OFStatsType statsType){
			this.statsType=statsType;
			this.dpId=dpId;
			this.replies=null;
		}
		public List<OFStatsReply> getReplies() {
			return replies;
		}
		public DatapathId getDpId() {
			return dpId;
		}
		@Override
		public void run(){
			replies=getSwitchStatistics(dpId, statsType);
		}
	}
	protected Map<DatapathId, List<OFStatsReply>> getSwitchStatistics(Set<DatapathId> dpIds , OFStatsType statsType){
		Map<DatapathId, List<OFStatsReply>> replies = new HashMap<>();
		List<GetStatisticsThread> activeThreads = new ArrayList<>(dpIds.size());
		List<GetStatisticsThread> pendingRemovalThreads = new ArrayList<>();
		GetStatisticsThread t;
		for(DatapathId dpId : dpIds){
			if(statsType.equals(OFStatsType.FLOW)&&staticSwId.contains(dpId)) continue;
			t = new GetStatisticsThread(dpId, statsType);
			activeThreads.add(t);
			t.start();
		}
		for(int sleepCycle = 0 ; sleepCycle < 2 ; sleepCycle++){
			for(GetStatisticsThread curThread : activeThreads){
				if(curThread.getState() == State.TERMINATED){
					replies.put(curThread.getDpId(), curThread.getReplies());
					pendingRemovalThreads.add(curThread);
				}
			}
			for(GetStatisticsThread curThread : pendingRemovalThreads){
				activeThreads.remove(curThread);
			}
			pendingRemovalThreads.clear();
			if(activeThreads.isEmpty()){
				break;
			}
			try{
				Thread.sleep(500);
			}
			catch(InterruptedException e){
				log.error("Interrupted while waiting for statistics", e);
			}
		}
		return replies;
	}
	@SuppressWarnings("unchecked")
	protected List<OFStatsReply> getSwitchStatistics(DatapathId swId , OFStatsType statsType){
		IOFSwitch sw=switchService.getSwitch(swId);
		ListenableFuture<?> future;
		List<OFStatsReply> values=null;
		Match match;
		if(sw!=null){
			OFStatsRequest<?> request = null;
			switch(statsType){
			case FLOW:
				match = sw.getOFFactory().buildMatch().build();
				request = sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) == 0 
						? sw.getOFFactory().buildFlowStatsRequest().setMatch(match).setOutPort(OFPort.ANY)
						.setTableId(DYNAMIC_TABLE_ID).build()
						: sw.getOFFactory().buildFlowStatsRequest().setMatch(match).setOutPort(OFPort.ANY)
						.setTableId(DYNAMIC_TABLE_ID).setOutGroup(OFGroup.ANY).build();
				break;
			case AGGREGATE:
				match = sw.getOFFactory().buildMatch().build();
				request = sw.getOFFactory().buildAggregateStatsRequest().setMatch(match).setOutPort(OFPort.ANY)
						.setTableId(TableId.ALL).build();
				break;
			case PORT:
				request = sw.getOFFactory().buildPortStatsRequest().setPortNo(OFPort.ANY).build();
				break;
			case QUEUE:
				request = sw.getOFFactory().buildQueueStatsRequest().setPortNo(OFPort.ANY)
						.setQueueId(UnsignedLong.MAX_VALUE.longValue()).build();
				break;
			case DESC:
				request = sw.getOFFactory().buildDescStatsRequest().build();
				break;
			case GROUP:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) > 0) {
					request = sw.getOFFactory().buildGroupStatsRequest().build();
				}
				break;
			case METER:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) >= 0) {
					request = sw.getOFFactory().buildMeterStatsRequest().setMeterId(OFMeterSerializerVer13.ALL_VAL).build();
				}
				break;
			case GROUP_DESC:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) > 0) {
					request = sw.getOFFactory().buildGroupDescStatsRequest().build();
				}
				break;
			case GROUP_FEATURES:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) > 0) {
					request = sw.getOFFactory().buildGroupFeaturesStatsRequest().build();
				}
				break;
			case METER_CONFIG:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) >= 0) {
					request = sw.getOFFactory().buildMeterConfigStatsRequest().build();
				}
				break;
			case METER_FEATURES:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) >= 0) {
					request = sw.getOFFactory().buildMeterFeaturesStatsRequest().build();
				}
				break;
			case TABLE:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) > 0) {
					request = sw.getOFFactory().buildTableStatsRequest().build();
				}
				break;
			case TABLE_FEATURES:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_10) > 0) {
					request = sw.getOFFactory().buildTableFeaturesStatsRequest().build();
				}
				break;
			case PORT_DESC:
				if (sw.getOFFactory().getVersion().compareTo(OFVersion.OF_13) >= 0) {
					request = sw.getOFFactory().buildPortDescStatsRequest().build();
				}
				break;
			case EXPERIMENTER:
			default:
				log.error("Stats Request Type {} not implemented yet", statsType.name());
				break;
			}
			try{
				if(request!=null){
					future = sw.writeStatsRequest(request);
					values = (List<OFStatsReply>) future.get(500, TimeUnit.MILLISECONDS);
				}
			}
			catch(Exception e){
				log.error("Failure retrieving statistics from switch {}. {}", sw, e);
			}
		}
		return values;
	}
	private class PortPacketsCollector implements Runnable {
		@Override
		public void run() {
			Map<DatapathId, List<OFStatsReply>> replies = getSwitchStatistics(portsInBind.keySet(),OFStatsType.PORT);
			for (Entry<DatapathId, List<OFStatsReply>> e : replies.entrySet()) {
				U64 maxInPacketNum=U64.ZERO;
				for (OFStatsReply r : e.getValue()) {
					OFPortStatsReply psr = (OFPortStatsReply) r;
					for (OFPortStatsEntry pse : psr.getEntries()) {
						SwitchPort sp = new SwitchPort(e.getKey(), pse.getPortNo());
						U64 prePacketsNum;
						if (inPortPackets.containsKey(sp) || inTentativePortPackets.containsKey(sp)) {
								prePacketsNum = inPortPackets.get(sp);
								prePacketsNum = inTentativePortPackets.get(sp);
								inTentativePortPackets.remove(sp);
							} else {
								log.error("Inconsistent state between tentative and official port stats lists.");
								return;
							}
							U64 inPacketsCounted = pse.getRxPackets();
							U64 inPacketsNum=inPacketsCounted.subtract(prePacketsNum);
							inPortPackets.put(sp,inPacketsCounted);
							inPortPacketsRes.put(sp, U64.of((long)(inPacketsNum.getValue()/0.7)));
							inTentativePortPackets.put(sp, pse.getRxPackets());
						}	
						if (outPortPackets.containsKey(sp) || outTentativePortPackets.containsKey(sp)) {
								prePacketsNum = outPortPackets.get(sp);
								prePacketsNum = outTentativePortPackets.get(sp);
								outTentativePortPackets.remove(sp);
							} else {
								log.error("Inconsistent state between tentative and official port stats lists.");
								return;
							}
							U64 outPacketsCounted = pse.getTxPackets();
							U64 outPacketsNum=outPacketsCounted.subtract(prePacketsNum);
							outPortPackets.put(sp,outPacketsCounted);
							outPortPacketsRes.put(sp, U64.of((long)(outPacketsNum.getValue()/0.7)));
							outTentativePortPackets.put(sp, pse.getTxPackets());
						}
						if(rank.containsKey(sp)) {
							if(timeToSave) {
								packetsInRecord.put(sp, pse.getRxPackets());
							}
							if(cycleTime==0) {
								portsList.add(sp);
								packetsInPeriod.put(sp, pse.getRxPackets().subtract(packetsInRecord.get(sp)));
								if(packetsInPeriod.get(sp).compareTo(maxInPacketNum)>0) {
									maxInPacketNum=packetsInPeriod.get(sp);
								}
							}
						}
					}
				}
				if(cycleTime==0&&maxInPacketNum.getValue()!=0) {
					if(d!=1) {
						for(SwitchPort sp : portsList) {
							U64 u=packetsInPeriod.get(sp);
							int priority=u.getValue()/d>BINDING_PRIORITY?(int)(u.getValue()/d):BINDING_PRIORITY;
							rank.put(sp, priority);
						}
					}
				}
				portsList.clear();
			}
			if(timeToSave) {
				timeToSave=false;
			}
			if(cycleTime==0) {
				timeToSave=true;
				cycleTime=period;
				doFlowMod(new HashSet<>(abnormalPorts));
			}
			cycleTime--;
		}
	}
	@Override
	public U64 getInPacketsNum(DatapathId dpid, OFPort p) {
		return inPortPacketsRes.getOrDefault(new SwitchPort(dpid, p), U64.ZERO);
	}
	@Override
	public U64 getOutPacketsNum(DatapathId dpid, OFPort p) {
		return outPortPacketsRes.getOrDefault(new SwitchPort(dpid, p), U64.ZERO);
	}
	@Override
	public PacketOfFlow getPacketOfFlow(DatapathId dpid, OFPort p) {
		return packetOfFlows.get(new SwitchPort(dpid, p));
	}
	@Override
	public Object getAllPacketOfFlow() {
		List<Map.Entry<SwitchPort, PacketOfFlow>> temp = 
				new ArrayList<>(packetOfFlows.entrySet());
		Collections.sort(temp, new Comparator<Map.Entry<SwitchPort, PacketOfFlow>>() {
			@Override
			public int compare(Entry<SwitchPort, PacketOfFlow> o1, Entry<SwitchPort, PacketOfFlow> o2) {
				double cmp = o1.getValue().getDropRate() - o2.getValue().getDropRate();
				if(cmp < 0 ) 
					return 1;
				else if(cmp > 0)
					return -1;
				else
					return 0;
			}
		});
		return temp;
	}
	public Object getPortSet() {
		Map<String, String> temp = new HashMap<>();
		temp.put("normal", getPorts(normalPorts));
		temp.put("polling", getPorts(pickFromNormal));
		Set<SwitchPort> set=new HashSet<>(observePorts.keySet());
		set.removeAll(pickFromNormal);
		temp.put("observe", getPorts(set));
		temp.put("abnormal", getPorts(abnormalPorts));
		return temp;
	}
	public String getPorts(Collection<SwitchPort> ports){
		StringBuffer sb = new StringBuffer();
		for(SwitchPort port : ports) {
			sb.append(port.getSwitchDPID().toString() + "-" +port.getPort().getPortNumber() + ",");
		}
		if(!sb.toString().equals("")){
			sb.deleteCharAt(sb.lastIndexOf(","));
		}
		return sb.toString();
	}
	@Override
	public synchronized void changePlanByRest(int flag) {
		STATUS = flag;
	}
	@Override
	public Object showMaxTraffic() {
		Map<Integer, Double> map = new HashMap<>();
		for(Map.Entry<SwitchPort, U64> entry : inPortPacketsRes.entrySet()){
			SwitchPort sp = entry.getKey();
			if(rank.containsKey(sp)){
				int terminatorNum = computeTerminatorNum(sp);
				map.put(terminatorNum, curIn);
			}
		}
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("%-8s", "curIn:"));
		StringBuffer sb2 = new StringBuffer();
		sb2.append(String.format("%-8s", "maxIn:"));
		DecimalFormat df = new DecimalFormat("####0.00");
		for(Map.Entry<Integer, Double> entry : map.entrySet()){
			sb.append(String.format("%-12s", df.format(entry.getValue()) + "(" + entry.getKey() + ")"));
			sb2.append(String.format("%-12s", df.format(maxTraffics.get(entry.getKey())) + "(" + entry.getKey()  + ")"));
		}
		log.info("入流量、峰值比对：\r\n" + sb.toString() + "\r\n" + sb2.toString());
		return maxTraffics;
	}
	@Override
	public synchronized void updateMaxTraffic() {
		for(Map.Entry<SwitchPort, U64> entry : inPortPacketsRes.entrySet()){
			SwitchPort sp = entry.getKey();
			if(rank.containsKey(sp)){
				int terminatorNum = computeTerminatorNum(sp);
				if(curIn > maxTraffics.get(terminatorNum)){
					maxTraffics.put(terminatorNum, curIn);
				}
			}
		}
		StringBuffer sb = new StringBuffer();
		for(Map.Entry<Integer, Double> entry : maxTraffics.entrySet()){
			sb.append(entry.getKey() + " " + entry.getValue() + "\r\n");
		}
		log.info("======rest更新maxTraffic及峰值文件======" + "\r\n" + sb.toString());
		writeToTxt(filePath, false, sb.toString());
	}
	private void convert(DatapathId dpid) {
		if(convertFlag.get(dpid)&&dynamicRuleNumber.get(dpid)>staticRuleNumber.get(dpid)) {
			if((System.currentTimeMillis()-stableTime)/1000<20) return ;
			convertTable(dpid, true);
		}else if(!convertFlag.get(dpid)&&dynamicRuleNumber.get(dpid)<staticRuleNumber.get(dpid)) {
			convertTable(dpid, false);
		}
	}
	private void convertTable(DatapathId dpid, boolean toStatic) {
		if (toStatic) {
			convertFlag.put(dpid, false);
			List<Binding<?>> bindings=saviProvider.getBindings();
			Set<SwitchPort> switchPorts=new HashSet<>();
			for(Binding<?> binding : bindings) { 
				if(binding.getSwitchPort().getSwitchDPID().equals(dpid)) {
					switchPorts.add(binding.getSwitchPort());
				}
			}
			doFlowMod(switchPorts,STATIC_TABLE_ID);
			saviProvider.convertTable(dpid, true);
		} else {
			convertFlag.put(dpid, true);
			saviProvider.convertTable(dpid, false);
		}
	}
	protected int computeTerminatorNum(SwitchPort swport) {
		return saviProvider.getHostWithPort().get(swport);
	}
	@Override
	public void setPriorityLevel(int priorityLevel) {
		this.priorityLevel=priorityLevel;
	}
	@Override
	public Object getHostsCredit() {
		Map<Integer, Integer> map=new HashMap<>();
		for(Entry<SwitchPort, Integer> entry : hostsCredit.entrySet()) {
			int t=saviProvider.getHostWithPort().get(entry.getKey());
			map.put(t, entry.getValue()/8+1);
		}
		return map;
	}
	@Override
	public void setAutoCheck(boolean autoCheck) {
		this.autoCheck=autoCheck;
	}
	private void writeErrorLog(SwitchPort sp, boolean isAbnormal) {
		String text="";
		if (isAbnormal) {
			long lossNum=packetOfFlows.get(sp).getDropNum();
			if(pickFromNormal.contains(sp)) 
				lossNum=(long) (lossNum/0.28);
			text=sdflog.format(System.currentTimeMillis())+"  端口："+"SwitchPort [switchDPID=" + sp.getSwitchDPID().toString() +
		               ", port=" + sp.getPort()+"  主机："+saviProvider.getHostWithPort().get(sp) + "发现异常---" +"发包："+(packetOfFlows.get(sp).getPassNum()+packetOfFlows.get(sp).getDropNum())+"  丢包率："+packetOfFlows.get(sp).getDropRate()
		               +"  丢包数："+packetOfFlows.get(sp).getDropNum();
		} else {
			text=sdflog.format(System.currentTimeMillis())+"  端口："+"SwitchPort [switchDPID=" + sp.getSwitchDPID().toString() +
		               ", port=" + sp.getPort()+"  主机："+saviProvider.getHostWithPort().get(sp) + "恢复正常---" + "发包："+(packetOfFlows.get(sp).getPassNum()+packetOfFlows.get(sp).getDropNum())+"  丢包率："+packetOfFlows.get(sp).getDropRate()
		               +"  丢包数："+packetOfFlows.get(sp).getDropNum();
		}
		try {
			synchronized (filePath5) {
				File file =new File(filePath5);
				if(!file.exists()) {
					file.createNewFile();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		BufferedWriter bw=null;
		try {
			bw = new BufferedWriter(new FileWriter(filePath5,true));
			bw.append(text);
			bw.newLine();
			bw.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
			try {
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	@Override
	public int calculateRule(DatapathId dpid) {
		return staticSwId.contains(dpid)?staticRuleNumber.get(dpid):dynamicRuleNumber.get(dpid);
	}
	@Override
	public Map<DatapathId, Integer> calculateRule() {
		ruleNum.putAll(dynamicRuleNumber);
		if(staticSwId.isEmpty()) return ruleNum;
		for(DatapathId sw : staticSwId)	ruleNum.put(sw, staticRuleNumber.get(sw));
		return ruleNum;
	}
	@Override
	public String getFilePath() {
		return filePath5;
	}
}
