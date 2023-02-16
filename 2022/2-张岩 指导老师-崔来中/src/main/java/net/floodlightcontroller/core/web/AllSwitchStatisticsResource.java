package net.floodlightcontroller.core.web;
import java.lang.Thread.State;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
import org.projectfloodlight.openflow.protocol.OFStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsType;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.resource.Get;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class AllSwitchStatisticsResource extends SwitchResourceBase {
	protected static Logger log =
			LoggerFactory.getLogger(AllSwitchStatisticsResource.class);
	@Get("json")
	public Map<String, StatsReply> retrieve() {
		String statType = (String) getRequestAttributes().get(CoreWebRoutable.STR_STAT_TYPE);
		return retrieveInternal(statType);
	}
	private Map<String, StatsReply> retrieveInternal(String statType) {
		HashMap<String, StatsReply> model = new HashMap<String, StatsReply>();
		OFStatsType type = null;
		REQUESTTYPE rType = null;
		switch (statType) {
		case OFStatsTypeStrings.PORT:
			type = OFStatsType.PORT;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.PORT_DESC:
			type = OFStatsType.PORT_DESC;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.QUEUE:
			type = OFStatsType.QUEUE;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.FLOW:
			type = OFStatsType.FLOW;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.AGGREGATE:
			type = OFStatsType.AGGREGATE;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.DESC:
			type = OFStatsType.DESC;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.TABLE:
			type = OFStatsType.TABLE;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.TABLE_FEATURES:
			type = OFStatsType.TABLE_FEATURES;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.GROUP:
			type = OFStatsType.GROUP;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.GROUP_DESC:        	
			type = OFStatsType.GROUP_DESC;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.GROUP_FEATURES:        	
			type = OFStatsType.GROUP_FEATURES;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.METER:
			type = OFStatsType.METER;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.METER_CONFIG:        	
			type = OFStatsType.METER_CONFIG;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.METER_FEATURES:        	
			type = OFStatsType.METER_FEATURES;
			rType = REQUESTTYPE.OFSTATS;
			break;
		case OFStatsTypeStrings.FEATURES:
			rType = REQUESTTYPE.OFFEATURES;
			break;
		default:
			return model;
		}
		IOFSwitchService switchService = (IOFSwitchService) getContext().getAttributes().
				get(IOFSwitchService.class.getCanonicalName());
		Set<DatapathId> switchDpids = switchService.getAllSwitchDpids();
		List<GetConcurrentStatsThread> activeThreads = new ArrayList<GetConcurrentStatsThread>(switchDpids.size());
		List<GetConcurrentStatsThread> pendingRemovalThreads = new ArrayList<GetConcurrentStatsThread>();
		GetConcurrentStatsThread t;
		for (DatapathId l : switchDpids) {
			t = new GetConcurrentStatsThread(l, rType, type);
			activeThreads.add(t);
			t.start();
		}
		for (int iSleepCycles = 0; iSleepCycles < 12; iSleepCycles++) {
			for (GetConcurrentStatsThread curThread : activeThreads) {
				if (curThread.getState() == State.TERMINATED) {
					if (rType == REQUESTTYPE.OFSTATS) {
						model.put(curThread.getSwitchId().toString(), new StatsReply(curThread.getSwitchId(), curThread.getStatisticsReply(), type));
					} else if (rType == REQUESTTYPE.OFFEATURES) {
						model.put(curThread.getSwitchId().toString(), new StatsReply(curThread.getSwitchId(), curThread.getFeaturesReply(), type));
					}
					pendingRemovalThreads.add(curThread);
				}
			}
			for (GetConcurrentStatsThread curThread : pendingRemovalThreads) {
				activeThreads.remove(curThread);
			}
			pendingRemovalThreads.clear();
			if (activeThreads.isEmpty()) {
				break;
			}
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				log.error("Interrupted while waiting for statistics", e);
			}
		}
		return model;
	}
	protected class GetConcurrentStatsThread extends Thread {
		private List<OFStatsReply> switchReply;
		private DatapathId switchId;
		private OFStatsType statType;
		private REQUESTTYPE requestType;
		private OFFeaturesReply featuresReply;
		public GetConcurrentStatsThread(DatapathId switchId, REQUESTTYPE requestType, OFStatsType statType) {
			this.switchId = switchId;
			this.requestType = requestType;
			this.statType = statType;
			this.switchReply = null;
			this.featuresReply = null;
		}
		public List<OFStatsReply> getStatisticsReply() {
			return switchReply;
		}
		public OFFeaturesReply getFeaturesReply() {
			return featuresReply;
		}
		public DatapathId getSwitchId() {
			return switchId;
		}
		@Override
		public void run() {
			if ((requestType == REQUESTTYPE.OFSTATS) && (statType != null)) {
				switchReply = getSwitchStatistics(switchId, statType);
			} else if (requestType == REQUESTTYPE.OFFEATURES) {
				featuresReply = getSwitchFeaturesReply(switchId);
			}
		}
	}
}
