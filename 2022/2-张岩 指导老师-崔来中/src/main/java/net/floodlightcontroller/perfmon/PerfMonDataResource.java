package net.floodlightcontroller.perfmon;
import org.restlet.data.Status;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class PerfMonDataResource extends ServerResource {
    protected static Logger logger = LoggerFactory.getLogger(PerfMonDataResource.class);  
    @Get("json")
    public CumulativeTimeBucket handleApiQuery() {        
        IPktInProcessingTimeService pktinProcTime = 
            (IPktInProcessingTimeService)getContext().getAttributes().
                get(IPktInProcessingTimeService.class.getCanonicalName());
        setStatus(Status.SUCCESS_OK, "OK");
        if (!pktinProcTime.isEnabled()){
        	pktinProcTime.setEnabled(true);
        	logger.warn("Requesting performance monitor data when performance monitor is disabled. Turning it on");
        }
        if (pktinProcTime.isEnabled()) {
            CumulativeTimeBucket ctb = pktinProcTime.getCtb();
            ctb.computeAverages();
            return ctb;
        }
        return null;
    }
}