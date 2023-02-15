package net.floodlightcontroller.core.web;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.web.StatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsType;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.resource.Get;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class SwitchStatisticsResource extends SwitchResourceBase {
	protected static Logger log = 
			LoggerFactory.getLogger(SwitchStatisticsResource.class);
	@Get("json")
	public StatsReply retrieve(){
		StatsReply result = new StatsReply();
		String switchIdStr = (String) getRequestAttributes().get(CoreWebRoutable.STR_SWITCH_ID);
		DatapathId switchId;
		String statType = (String) getRequestAttributes().get(CoreWebRoutable.STR_STAT_TYPE);
		IOFSwitchService switchService = (IOFSwitchService) getContext().getAttributes().
				get(IOFSwitchService.class.getCanonicalName());
		try {
			switchId = DatapathId.of(switchIdStr);
		}
		if (!switchId.equals(DatapathId.NONE) && switchService.getSwitch(switchId) != null) {			
			switch (statType) {
			case OFStatsTypeStrings.PORT:
				values = getSwitchStatistics(switchId, OFStatsType.PORT);
				result.setStatType(OFStatsType.PORT);
				break;
			case OFStatsTypeStrings.QUEUE:
				values = getSwitchStatistics(switchId, OFStatsType.QUEUE);
				result.setStatType(OFStatsType.QUEUE);
				break;
			case OFStatsTypeStrings.FLOW:
				values = getSwitchStatistics(switchId, OFStatsType.FLOW);
				result.setStatType(OFStatsType.FLOW);
				break;
			case OFStatsTypeStrings.AGGREGATE:
				values = getSwitchStatistics(switchId, OFStatsType.AGGREGATE);
				result.setStatType(OFStatsType.AGGREGATE);
				break;
			case OFStatsTypeStrings.DESC:
				values = getSwitchStatistics(switchId, OFStatsType.DESC);
				result.setStatType(OFStatsType.DESC);
				break;			
			case OFStatsTypeStrings.PORT_DESC:
				values = getSwitchStatistics(switchId, OFStatsType.PORT_DESC);
				result.setStatType(OFStatsType.PORT_DESC);
				break;
			case OFStatsTypeStrings.GROUP:
				values = getSwitchStatistics(switchId, OFStatsType.GROUP);
				result.setStatType(OFStatsType.GROUP);
				break;
			case OFStatsTypeStrings.GROUP_DESC:
				values = getSwitchStatistics(switchId, OFStatsType.GROUP_DESC);
				result.setStatType(OFStatsType.GROUP_DESC);
				break;
			case OFStatsTypeStrings.GROUP_FEATURES:
				values = getSwitchStatistics(switchId, OFStatsType.GROUP_FEATURES);
				result.setStatType(OFStatsType.GROUP_FEATURES);
				break;
			case OFStatsTypeStrings.METER:
				values = getSwitchStatistics(switchId, OFStatsType.METER);
				result.setStatType(OFStatsType.METER);
				break;
			case OFStatsTypeStrings.METER_CONFIG:
				values = getSwitchStatistics(switchId, OFStatsType.METER_CONFIG);
				result.setStatType(OFStatsType.METER_CONFIG);
				break;
			case OFStatsTypeStrings.METER_FEATURES:
				values = getSwitchStatistics(switchId, OFStatsType.METER_FEATURES);
				result.setStatType(OFStatsType.METER_FEATURES);
				break;
			case OFStatsTypeStrings.TABLE:
				values = getSwitchStatistics(switchId, OFStatsType.TABLE);
				result.setStatType(OFStatsType.TABLE);
				break;
			case OFStatsTypeStrings.TABLE_FEATURES:
				values = getSwitchStatistics(switchId, OFStatsType.TABLE_FEATURES);
				result.setStatType(OFStatsType.TABLE_FEATURES);
				break;
			case OFStatsTypeStrings.EXPERIMENTER:
				values = getSwitchFeaturesReply(switchId);
				result.setStatType(OFStatsType.EXPERIMENTER);
				break;
			case OFStatsTypeStrings.FEATURES:
				values = getSwitchFeaturesReply(switchId);
			default:
				log.error("Invalid or unimplemented stat request type {}", statType);
				break;
			}
		} else {
			log.error("Invalid or disconnected switch {}", switchIdStr);
		}
		result.setDatapathId(switchId);
		return result;
	}
}
