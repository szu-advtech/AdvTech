package net.floodlightcontroller.core.web;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import net.floodlightcontroller.debugcounter.DebugCounterResource;
import org.restlet.resource.Get;
public class CounterResource extends CounterResourceBase {
    @Get("json")
    public Map<String, Object> retrieve() {
        String counterTitle = (String) getRequestAttributes().get(CoreWebRoutable.STR_CTR_TITLE);
        String counterModule = (String) getRequestAttributes().get(CoreWebRoutable.STR_CTR_MODULE);
        Map<String, Object> model = new HashMap<String, Object>();
        long dc;
            List<DebugCounterResource> counters = this.debugCounterService.getAllCounterValues();
            if (counters != null) {
                Iterator<DebugCounterResource> it = counters.iterator();
                while (it.hasNext()) {
                    DebugCounterResource dcr = it.next();
                    String counterName = dcr.getCounterHierarchy();
                    dc = dcr.getCounterValue();
                    model.put(counterName, dc);
                }   
            }   
            List<DebugCounterResource> counters = this.debugCounterService.getModuleCounterValues(counterModule);
            if (counters != null) {
                Iterator<DebugCounterResource> it = counters.iterator();
                while (it.hasNext()) {
                    DebugCounterResource dcr = it.next();
                    String counterName = dcr.getCounterHierarchy();
                    dc = dcr.getCounterValue();
                    model.put(counterName, dc);
                }   
            }   
            List<DebugCounterResource> counters = this.debugCounterService.getCounterHierarchy(counterModule, counterTitle);
            if (counters != null) {
                Iterator<DebugCounterResource> it = counters.iterator();
                while (it.hasNext()) {
                    DebugCounterResource dcr = it.next();
                    String counterName = dcr.getCounterHierarchy();
                    dc = dcr.getCounterValue();
                    model.put(counterName, dc);
                }   
            }   
        }
        return model;
    }
}
