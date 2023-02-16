package net.floodlightcontroller.core.web;
import java.util.Map;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import net.floodlightcontroller.core.IFloodlightProviderService;
public class ControllerSummaryResource extends ServerResource {
    @Get("json")
    public Map<String, Object> retrieve() {
        IFloodlightProviderService floodlightProvider = 
            (IFloodlightProviderService)getContext().getAttributes().
                get(IFloodlightProviderService.class.getCanonicalName());
        return floodlightProvider.getControllerInfo("summary");
    }
}
