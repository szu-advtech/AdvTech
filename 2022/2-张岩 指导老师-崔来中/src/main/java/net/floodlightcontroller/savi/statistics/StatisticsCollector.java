package net.floodlightcontroller.savi.statistics;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sun.management.OperatingSystemMXBean;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.savi.statistics.web.KeyValuePair;
import net.floodlightcontroller.savi.statistics.web.KeyValuePairSet;
import net.floodlightcontroller.savi.statistics.web.StatisticsWebRoutable;
import net.floodlightcontroller.threadpool.IThreadPoolService;
public class StatisticsCollector implements IFloodlightModule, IOFMessageListener, IStatisticsCollector {
	protected static Logger log = LoggerFactory.getLogger(StatisticsCollector.class);
	protected static final int STATISTICS_APP_ID = 123456;
	public static Map<OFType, String> STATISTICS_RECORD_TYPES = new HashMap<>();
	static {
		STATISTICS_RECORD_TYPES = new HashMap<>();
		STATISTICS_RECORD_TYPES.put(OFType.FLOW_MOD, "FLOW_MOD");
		STATISTICS_RECORD_TYPES.put(OFType.PACKET_IN, "PACKET_IN");
		STATISTICS_RECORD_TYPES.put(OFType.PACKET_OUT, "PACKET_OUT");
		STATISTICS_RECORD_TYPES.put(OFType.FLOW_REMOVED, "FLOW_REMOVED");
		STATISTICS_RECORD_TYPES.put(OFType.ECHO_REPLY, "ECHO_REPLAY");
		STATISTICS_RECORD_TYPES.put(OFType.ECHO_REQUEST, "ECHO_REQUEST");
		STATISTICS_RECORD_TYPES.put(OFType.ERROR, "ERROR");
		STATISTICS_RECORD_TYPES.put(OFType.FEATURES_REPLY, "FEATURE_REPLAY");
		STATISTICS_RECORD_TYPES.put(OFType.FEATURES_REQUEST, "FEATURE_REQUEST");
		STATISTICS_RECORD_TYPES.put(OFType.METER_MOD, "METER_MOD");
	}
	protected static final int UPDATE_TASK_INTERVAl = 100; 
	protected static int COLLECTOR_TASK_INTERVAL = 10;
	protected SampleValue cpuSmaple;
	protected SampleValue memorySample;
	protected Map<OFType, ReferenceInterger> collector;
	protected Queue<OFMessage> updateQueue;
	protected IFloodlightProviderService floodlightProviderService;
	protected IThreadPoolService threadPoolService;
	protected IRestApiService restApiService;
	protected int statisticsTimer = 0;
	protected SingletonTask updateTask;
	@Override
	public String getName() {
		return "statictics-collector";
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return false;
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		return false;
	}
	protected void print(int id,OFType type, ReferenceInterger i){
		log.info("STATISTICS "+id+" "+STATISTICS_RECORD_TYPES.get(type)+" "+i.getAverage()+" "+i.getSum());
	}
	private boolean filter(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		return true;
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg,
			FloodlightContext cntx) {
		if(filter(sw, msg, cntx)) {
			updateQueue.add(msg);
		}
		return Command.CONTINUE;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IStatisticsCollector.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		Map<Class<? extends IFloodlightService>, IFloodlightService> m =
				new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
		m.put(IStatisticsCollector.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IThreadPoolService.class);
		l.add(IRestApiService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		this.floodlightProviderService 	= context.getServiceImpl(IFloodlightProviderService.class);
		this.threadPoolService 			= context.getServiceImpl(IThreadPoolService.class);
		this.restApiService				= context.getServiceImpl(IRestApiService.class);
		this.collector 		= new ConcurrentHashMap<>();
		this.updateQueue 	= new ConcurrentLinkedDeque<>();
		this.cpuSmaple 		= new SampleValue();
		this.memorySample 	= new SampleValue();
		for(OFType type:STATISTICS_RECORD_TYPES.keySet()) {
			this.collector.put(type, new ReferenceInterger());
		}
	}
	@Override
	public void startUp(FloodlightModuleContext context) throws FloodlightModuleException {
		for(OFType type:STATISTICS_RECORD_TYPES.keySet()) {
			floodlightProviderService.addOFMessageListener(type, this);
		}
		final Runtime runtime = Runtime.getRuntime();
		final OperatingSystemMXBean operatingSystemMXBean = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);
		restApiService.addRestletRoutable(new StatisticsWebRoutable());
		ScheduledExecutorService ses = threadPoolService.getScheduledExecutor();
		updateTask = new SingletonTask(ses, new Runnable() {
			int counter = 0;
			@Override
			public void run() {
				while(!updateQueue.isEmpty()) {
					OFType type = updateQueue.remove().getType();
					if(STATISTICS_RECORD_TYPES.containsKey(type)) {
						collector.get(type).plusOne();
					}
				}
				if(counter == 0) {
					for(OFType type:STATISTICS_RECORD_TYPES.keySet()) {
						collector.get(type).clear();
					}
					statisticsTimer ++;
				}
				counter = (counter + 1) % COLLECTOR_TASK_INTERVAL;
				cpuSmaple.add(operatingSystemMXBean.getProcessCpuLoad());
				memorySample.add(runtime.totalMemory()-runtime.freeMemory());
				if(counter == 0) {
					cpuSmaple.clear();
					memorySample.clear();				
				}
				updateTask.reschedule(UPDATE_TASK_INTERVAl, TimeUnit.MILLISECONDS);
			}
		});
		updateTask.reschedule(UPDATE_TASK_INTERVAl, TimeUnit.MILLISECONDS);
	}
	@Override
	public KeyValuePairSet getStatistics() {
		KeyValuePairSet pairs = new KeyValuePairSet();
		for(OFType type: STATISTICS_RECORD_TYPES.keySet()) {
			pairs.add(new KeyValuePair(STATISTICS_RECORD_TYPES.get(type), ""+collector.get(type).getLastAverage()));
		}
		pairs.add(new KeyValuePair("CPU", ""+cpuSmaple.getAverage()));
		pairs.add(new KeyValuePair("MEMORY", ""+memorySample.getAverage()));
		pairs.add(new KeyValuePair("TIME", ""+statisticsTimer));
		return pairs;
	}
}
class SampleValue {
	double average;
	double sum;
	int counter = 0;
	public SampleValue() {
		average = 0.0;
		sum = 0.0;
		counter = 0;
	}
	public void add(double value) {
		sum += value;
		counter ++;
	}
	public void clear() {
		if(counter == 0) {
			average = 0;
		}
		else {
			average = sum / counter;
		}
		sum = 0;
		counter = 0;
	}
	public double getAverage() {
		return average;
	}
	@Override
	public String toString() {
		return "average:"+average+" sum:"+sum;
	}
}
class ReferenceInterger {
	int average;
	int lastAverage = 0;
	int sum;
	public ReferenceInterger(){
		average = 0;
		sum = 0;
	}
	public int getAverage() {
		return average;
	}
	public int getLastAverage() {
		return lastAverage;
	}
	public int getSum() {
		return sum;
	}
	public void clear() {
		lastAverage = average;
		average = 0;
	}
	public void plusOne() {
		sum += 1;
		average += 1;
	}
	@Override
	public String toString() {
		return "lastAverage:"+lastAverage+" average:"+average+" sum:"+sum;
	}
}