package net.floodlightcontroller.flowcache;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.internal.Controller;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.ListenerDispatcher;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.debugcounter.DebugCounterResource;
import net.floodlightcontroller.debugcounter.IDebugCounter;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.flowcache.FlowReconcileQuery.FlowReconcileQueryDebugEvent;
import net.floodlightcontroller.flowcache.IFlowReconcileListener;
import net.floodlightcontroller.flowcache.OFMatchReconcile;
import net.floodlightcontroller.flowcache.PriorityPendingQueue.EventPriority;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import org.projectfloodlight.openflow.protocol.OFType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
@Deprecated
public class FlowReconcileManager implements IFloodlightModule, IFlowReconcileService {
	private static Logger logger =  LoggerFactory.getLogger(FlowReconcileManager.class);
	protected IThreadPoolService threadPoolService;
	protected IDebugCounterService debugCounterService;
	protected IFloodlightProviderService floodlightProviderService;
	protected ListenerDispatcher<OFType, IFlowReconcileListener> flowReconcileListeners;
	PriorityPendingQueue <OFMatchReconcile> flowQueue;
	protected SingletonTask flowReconcileTask;
	protected DebugCounterResource ctrControllerPktIn;
	protected IDebugCounter lastPacketInCounter;
	protected final static int MAX_SYSTEM_LOAD_PER_SECOND = 10000;
	protected final static int MIN_FLOW_RECONCILE_PER_SECOND = 200;
	protected final static int FLOW_RECONCILE_DELAY_MILLISEC = 10;
	protected Date lastReconcileTime;
	protected static final String EnableConfigKey = "enable";
	public static final String PACKAGE = FlowReconcileManager.class.getPackage().getName();
	private IDebugCounter ctrFlowReconcileRequest;
	private IDebugCounter ctrReconciledFlows;
	protected boolean flowReconcileEnabled;
	private DebugCounterResource ctrPacketInRsrc = null;
	public AtomicInteger flowReconcileThreadRunCount;
	@Override
	public synchronized void addFlowReconcileListener(
			IFlowReconcileListener listener) {
		flowReconcileListeners.addListener(OFType.FLOW_MOD, listener);
		if (logger.isTraceEnabled()) {
			StringBuffer sb = new StringBuffer();
			sb.append("FlowMod listeners: ");
			for (IFlowReconcileListener l :
				flowReconcileListeners.getOrderedListeners()) {
				sb.append(l.getName());
				sb.append(",");
			}
			logger.trace(sb.toString());
		}
	}
	@Override
	public synchronized void removeFlowReconcileListener(
			IFlowReconcileListener listener) {
		flowReconcileListeners.removeListener(listener);
	}
	@Override
	public synchronized void clearFlowReconcileListeners() {
		flowReconcileListeners.clearListeners();
	}
	@Override
	public void reconcileFlow(OFMatchReconcile ofmRcIn, EventPriority priority) {
		if (ofmRcIn == null) return;
		OFMatchReconcile myOfmRc = new OFMatchReconcile(ofmRcIn);
		flowQueue.offer(myOfmRc, priority);
		ctrFlowReconcileRequest.increment();
		Date currTime = new Date();
		long delay = 0;
		if (currTime.after(new Date(lastReconcileTime.getTime() + 1000))) {
			delay = 0;
		} else {
			delay = FLOW_RECONCILE_DELAY_MILLISEC;
		}
		flowReconcileTask.reschedule(delay, TimeUnit.MILLISECONDS);
		if (logger.isTraceEnabled()) {
			logger.trace("Reconciling flow: {}, total: {}",
					myOfmRc.toString(), flowQueue.size());
		}
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFlowReconcileService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService>
	getServiceImpls() {
		Map<Class<? extends IFloodlightService>,
		IFloodlightService> m =
		new HashMap<Class<? extends IFloodlightService>,
		IFloodlightService>();
		m.put(IFlowReconcileService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>>
	getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IThreadPoolService.class);
		l.add(IDebugCounterService.class);
		return null;
	}
	@Override
	public void init(FloodlightModuleContext context)
			throws FloodlightModuleException {
		floodlightProviderService = context.getServiceImpl(IFloodlightProviderService.class);
		threadPoolService = context.getServiceImpl(IThreadPoolService.class);
		debugCounterService = context.getServiceImpl(IDebugCounterService.class);
		flowQueue = new PriorityPendingQueue<OFMatchReconcile>();
		flowReconcileListeners = new ListenerDispatcher<OFType, IFlowReconcileListener>();
		Map<String, String> configParam = context.getConfigParams(this);
		String enableValue = configParam.get(EnableConfigKey);
		registerFlowReconcileManagerDebugCounters();
		flowReconcileEnabled = true;
		if (enableValue != null &&
				enableValue.equalsIgnoreCase("false")) {
			flowReconcileEnabled = false;
		}
		flowReconcileThreadRunCount = new AtomicInteger(0);
		lastReconcileTime = new Date(0);
		logger.debug("FlowReconcile is {}", flowReconcileEnabled);
	}
	private void registerFlowReconcileManagerDebugCounters() throws FloodlightModuleException {
		if (debugCounterService == null) {
			logger.error("Debug Counter Service not found.");
		}
		try {
			debugCounterService.registerModule(PACKAGE);
			ctrFlowReconcileRequest = debugCounterService.registerCounter(PACKAGE, "flow-reconcile-request",
					"All flow reconcile requests received by this module");
			ctrReconciledFlows = debugCounterService.registerCounter(PACKAGE, "reconciled-flows",
					"All flows reconciled successfully by this module");
		} catch (Exception e) {
			throw new FloodlightModuleException(e.getMessage());
		}
	}
	@Override
	public void startUp(FloodlightModuleContext context) {
		ScheduledExecutorService ses = threadPoolService.getScheduledExecutor();
		flowReconcileTask = new SingletonTask(ses, new Runnable() {
			@Override
			public void run() {
				try {
					if (doReconcile()) {
						flowReconcileTask.reschedule(
								FLOW_RECONCILE_DELAY_MILLISEC,
								TimeUnit.MILLISECONDS);
					}
				} catch (Exception e) {
					logger.warn("Exception in doReconcile(): {}", e);
				}
			}
		});
		String packetInName = OFType.PACKET_IN.getClass().getName();
		packetInName = packetInName.substring(packetInName.lastIndexOf('.')+1);
	}
	protected void updateFlush() {
	}
	protected boolean doReconcile() {
		if (!flowReconcileEnabled) {
			return false;
		}
		lastReconcileTime = new Date();
		ArrayList<OFMatchReconcile> ofmRcList =
				new ArrayList<OFMatchReconcile>();
		int reconcileCapacity = getCurrentCapacity();
		if (logger.isTraceEnabled()) {
			logger.trace("Reconcile capacity {} flows", reconcileCapacity);
		}
		while (!flowQueue.isEmpty() && reconcileCapacity > 0) {
			OFMatchReconcile ofmRc = flowQueue.poll();
			reconcileCapacity--;
			if (ofmRc != null) {
				ofmRcList.add(ofmRc);
				ctrReconciledFlows.increment();
				if (logger.isTraceEnabled()) {
					logger.trace("Add flow {} to be the reconcileList", ofmRc.cookie);
				}
			} else {
				break;
			}
		}
		IFlowReconcileListener.Command retCmd;
		if (ofmRcList.size() > 0) {
			List<IFlowReconcileListener> listeners =
					flowReconcileListeners.getOrderedListeners();
			if (listeners == null) {
				if (logger.isTraceEnabled()) {
					logger.trace("No flowReconcile listener");
				}
				return false;
			}
			for (IFlowReconcileListener flowReconciler :
				flowReconcileListeners.getOrderedListeners()) {
				if (logger.isTraceEnabled())
				{
					logger.trace("Reconciling flow: call listener {}",
							flowReconciler.getName());
				}
				retCmd = flowReconciler.reconcileFlows(ofmRcList);
				if (retCmd == IFlowReconcileListener.Command.STOP) {
					break;
				}
			}
			for (OFMatchReconcile ofmRc : ofmRcList) {
				if (ofmRc.origReconcileQueryEvent != null) {
					ofmRc.origReconcileQueryEvent.evType.getDebugEvent()
					.newEventWithFlush(new FlowReconcileQueryDebugEvent(
							ofmRc.origReconcileQueryEvent,
							"Flow Reconciliation Complete",
							ofmRc));
				}
			}
			updateFlush();
			flowReconcileThreadRunCount.incrementAndGet();
		} else {
			if (logger.isTraceEnabled()) {
				logger.trace("No flow to be reconciled.");
			}
		}
		if (flowQueue.isEmpty()) {
			return false;
		} else {
			if (logger.isTraceEnabled()) {
				logger.trace("{} more flows to be reconciled.",
						flowQueue.size());
			}
			return true;
		}
	}
	protected int getCurrentCapacity() {
		List<DebugCounterResource> contCtrRsrcList = debugCounterService.getModuleCounterValues(Controller.class.getName());
		for (DebugCounterResource dcr : contCtrRsrcList) {
			if (dcr.getCounterHierarchy().equals("packet-in")) {
				ctrPacketInRsrc = dcr;
				break;
			}
		}
		if (ctrPacketInRsrc == null || ctrPacketInRsrc.getCounterValue() == null || ctrPacketInRsrc.getCounterValue() == 0) {
			logger.debug("counter {} doesn't exist", ctrPacketInRsrc);
			return minFlows;
		}
		if (lastPacketInCounter.getCounterValue() == 0) {
			logger.debug("First time get the count for {}", lastPacketInCounter);
			return minFlows;
		}
		int pktInRate = getPktInRate(ctrPacketInRsrc, new Date());
		lastPacketInCounter.reset();
		int capacity = minFlows;
		if ((pktInRate + MIN_FLOW_RECONCILE_PER_SECOND) <=
				MAX_SYSTEM_LOAD_PER_SECOND) {
			capacity = (MAX_SYSTEM_LOAD_PER_SECOND - pktInRate)
		}
		if (logger.isTraceEnabled()) {
			logger.trace("Capacity is {}", capacity);
		}
		return capacity;
	}
	protected int getPktInRate(DebugCounterResource newCnt, Date currentTime) {
		if (newCnt == null ||
				newCnt.getCounterLastModified() == null ||
				newCnt.getCounterValue() == null) {
			return 0;
		}
		if (newCnt.getCounterLastModified() < lastPacketInCounter.getLastModified()) {
			logger.debug("Time is going backward. new {}, old {}",
					newCnt.getCounterLastModified(),
					lastPacketInCounter.getLastModified());
			return MAX_SYSTEM_LOAD_PER_SECOND;
		}
		long elapsedTimeInSecond = (currentTime.getTime() - lastPacketInCounter.getLastModified()) / 1000;
		if (elapsedTimeInSecond == 0) {
			return 0;
		}
		long diff = 0;
		long newValue = newCnt.getCounterValue().longValue();
		long oldValue = lastPacketInCounter.getCounterValue();
		if (newValue < oldValue) {
			diff = Long.MAX_VALUE - oldValue + newValue;
		} else {
			diff = newValue - oldValue;
		}
		return (int)(diff/elapsedTimeInSecond);
	}
}
