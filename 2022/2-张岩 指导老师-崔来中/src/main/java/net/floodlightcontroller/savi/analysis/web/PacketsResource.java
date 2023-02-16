package net.floodlightcontroller.savi.analysis.web;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class PacketsResource extends ServerResource {
	@Get("json")
	public Object retrieve(){
		IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		IOFSwitchService switchService = (IOFSwitchService) getContext().getAttributes().get(IOFSwitchService.class.getCanonicalName());
		Map<String, PacketsNum> result = new HashMap<>();
		Set<DatapathId> dpids = switchService.getAllSwitchDpids();
		for(DatapathId dpid : dpids){
			IOFSwitch sw = switchService.getSwitch(dpid);
			Collection<OFPort> ports = sw.getEnabledPortNumbers();
			for(OFPort port : ports){
				result.put(dpid.toString() + "-" + port.getPortNumber(),
						new PacketsNum(analysisService.getInPacketsNum(dpid, port).getValue() + "",
								analysisService.getOutPacketsNum(dpid, port).getValue() + "", 0));
			}
		}
		return result;
	}
}
