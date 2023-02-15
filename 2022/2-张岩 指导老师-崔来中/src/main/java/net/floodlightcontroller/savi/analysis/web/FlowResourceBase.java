package net.floodlightcontroller.savi.analysis.web;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.projectfloodlight.openflow.protocol.OFStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsRequest;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.util.concurrent.ListenableFuture;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
public class FlowResourceBase extends ServerResource {
	protected static Logger log=LoggerFactory.getLogger(FlowResourceBase.class);
	@SuppressWarnings("unchecked")
	protected List<OFStatsReply> getSwitchStatistics(DatapathId switchId, TableId tableId){
		IOFSwitchService switchService= (IOFSwitchService) getContext().getAttributes().get(IOFSwitchService.class.getCanonicalName());
		IOFSwitch sw=switchService.getSwitch(switchId);
		ListenableFuture<?> feature;
		List<OFStatsReply> values=null;
		Match match=sw.getOFFactory().buildMatch().build();
		OFStatsRequest<?> req=sw.getOFFactory().buildFlowStatsRequest()								
				.setMatch(match)
				.setOutPort(OFPort.ANY)
				.setTableId(tableId)
				.build();
		try {
			if(req!=null) {
				feature=sw.writeStatsRequest(req);
				values=(List<OFStatsReply>) feature.get(10, TimeUnit.SECONDS);
			}
		} catch (Exception e) {
			log.error("Failure retrieving statistics from switch " + sw, e);
		}
		return values;
	}
	protected List<OFStatsReply> getSwitchStatistics(Set<DatapathId> sws, TableId tableId){
		List<OFStatsReply> values=new ArrayList<>();
		for(DatapathId switchId : sws) {
			values.addAll(getSwitchStatistics(switchId, tableId));
		}
		return values;
	}
}
