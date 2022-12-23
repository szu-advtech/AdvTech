package net.floodlightcontroller.core.web;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import org.restlet.resource.ResourceException;
import org.restlet.resource.ServerResource;
public class CounterResourceBase extends ServerResource {
    protected IDebugCounterService debugCounterService;
    @Override
    protected void doInit() throws ResourceException {
        super.doInit();
        debugCounterService = (IDebugCounterService) getContext().getAttributes().
                get(IDebugCounterService.class.getCanonicalName());
    }
}
