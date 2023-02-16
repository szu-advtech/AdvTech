package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class PacketOfFlowResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(PacketOfFlowResource.class);
	@Get("json")
	public Object retrieve(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		String d = (String) getRequestAttributes().get(AnalysisWebRoutable.DPID_STR);
		String p = (String) getRequestAttributes().get(AnalysisWebRoutable.PORT_STR);
		DatapathId dpid = DatapathId.NONE;
		if(!d.trim().equalsIgnoreCase("all")){
			try {
				dpid = DatapathId.of(d);
			} catch (Exception e) {
                log.error("Could not parse DPID {}", d);
                return Collections.singletonMap("Error", "Could not parse DPID " + d);
			}
		}
		OFPort port = OFPort.ALL;
		if(!p.trim().equalsIgnoreCase("all")){
			try {
				port = OFPort.of(Integer.parseInt(p));
			} catch (Exception e) {
				log.error("Could not parse PORT {}", d);
				return Collections.singletonMap("Error", "Could not parse PORT " + p);
			}
		}
		return analysisService.getPacketOfFlow(dpid, port);
	}
}
