package net.floodlightcontroller.core.web;
import java.util.HashMap;
import java.util.Map;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
public class ControllerMemoryResource extends ServerResource {
    @Get("json")
    public Map<String, Object> retrieve() {
        HashMap<String, Object> model = new HashMap<String, Object>();
        Runtime runtime = Runtime.getRuntime();
        model.put("total", new Long(runtime.totalMemory()));
        model.put("free", new Long(runtime.freeMemory()));
        return model;
    }
}
