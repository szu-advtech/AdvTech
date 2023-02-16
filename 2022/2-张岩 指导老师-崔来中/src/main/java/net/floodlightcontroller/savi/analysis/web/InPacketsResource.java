package net.floodlightcontroller.savi.analysis.web;
import java.util.Collections;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class InPacketsResource extends ServerResource {
	private static final Logger log = LoggerFactory.getLogger(InPacketsResource.class);
    @Get("json")
    public Object retrieve() {
        IAnalysisService analysisService = (IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
        String d = (String) getRequestAttributes().get(AnalysisWebRoutable.DPID_STR);
        String p = (String) getRequestAttributes().get(AnalysisWebRoutable.PORT_STR);
        DatapathId dpid = DatapathId.NONE;
        if (!d.trim().equalsIgnoreCase("all")) {
            try {
                dpid = DatapathId.of(d);
            } catch (Exception e) {
                log.error("Could not parse DPID {}", d);
                return Collections.singletonMap("ERROR", "Could not parse DPID " + d);
            }
        OFPort port = OFPort.ALL;
        if (!p.trim().equalsIgnoreCase("all")) {
            try {
                port = OFPort.of(Integer.parseInt(p));
            } catch (Exception e) {
                log.error("Could not parse port {}", p);
                return Collections.singletonMap("ERROR", "Could not parse port " + p);
            }
        }
        U64 outPacketsNum;
        outPacketsNum = analysisService.getOutPacketsNum(dpid, port);
        U64 inPacketsNum;
        inPacketsNum = analysisService.getInPacketsNum(dpid, port);
        return new PacketsNum(inPacketsNum.getValue() + "" , outPacketsNum.getValue() + "" , 0);
    }
}
class PacketsNum{
	String in;
	String out;
	int level;
	public int getLevel() {
		return level;
	}
	public void setLevel(int level) {
		this.level = level;
	}
	public String getIn() {
		return in;
	}
	public void setIn(String in) {
		this.in = in;
	}
	public String getOut() {
		return out;
	}
	public void setOut(String out) {
		this.out = out;
	}
	public PacketsNum(String in,String out,int level) {
		this.in = in;
		this.out = out;
		this.level = level;
	}
}
