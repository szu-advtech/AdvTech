package net.floodlightcontroller.savi.analysis.web;
import java.util.HashSet;
import java.util.Set;
import org.projectfloodlight.openflow.protocol.OFStatsType;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.TableId;
import org.restlet.resource.Get;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.web.StatsReply;
import net.floodlightcontroller.savi.service.SAVIProviderService;
public class FlowResource extends FlowResourceBase{
	Logger log=LoggerFactory.getLogger(FlowResource.class);
	@Get("json")
	public Set<StatsReply> retrieve() {
		Set<StatsReply> res=new HashSet<>();
		Object values=null;
		String tableIdStr=(String) getRequestAttributes().get(AnalysisWebRoutable.TABLE_ID);
		String switchIdStr=(String) getRequestAttributes().get(AnalysisWebRoutable.SWITCH_ID);
		DatapathId switchId=DatapathId.NONE;
		TableId tableId=TableId.NONE;
		SAVIProviderService saviProvider= (SAVIProviderService) getContext().getAttributes()
				.get(SAVIProviderService.class.getCanonicalName());
		if(switchIdStr.equalsIgnoreCase("all")) {
			switchId=DatapathId.NONE;
		}else {
			try {
				switchId=DatapathId.of(switchIdStr);
			} catch (Exception e) {
				log.info("解析交换机ID失败");
				return res;
			}
		}
		try {
			int id=Integer.parseInt(tableIdStr);
			tableId=TableId.of(id);
		} catch (Exception e) {
			log.info("解析tableId失败");
			return res;
		}
		if(switchId.equals(DatapathId.NONE)) {
			for(DatapathId swid : saviProvider.getPortsInBind().keySet()) {
					values=getSwitchStatistics(swid, tableId);
				StatsReply result=new StatsReply();
				result.setStatType(OFStatsType.FLOW);
				result.setDatapathId(swid);
				result.setValues(values);
				res.add(result);
			}
			return res;
		}
		values = getSwitchStatistics(switchId, tableId);
		StatsReply result=new StatsReply();
		result.setDatapathId(switchId);
		result.setValues(values);
		res.add(result);
		return res;
	}
}
