package net.floodlightcontroller.jython;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.internal.FloodlightProvider;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
public class JythonDebugInterface implements IFloodlightModule {
    protected static Logger log = LoggerFactory.getLogger(JythonDebugInterface.class);
    protected JythonServer debug_server;
    protected String jythonHost = null;
    protected int jythonPort = 6655;
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        return null;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
            getServiceImpls() {
        return null;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
            getModuleDependencies() {
        return null;
    }
    @Override
    public void init(FloodlightModuleContext context)
             throws FloodlightModuleException {
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
        Map<String, Object> locals = new HashMap<String, Object>();     
        for (Class<? extends IFloodlightService> s : context.getAllServices()) {
            String[] bits = s.getCanonicalName().split("\\.");
            String name = bits[bits.length-1];
            locals.put(name, context.getServiceImpl(s));
        }
        Map<String, String> configOptions = context.getConfigParams(this);
        jythonHost = configOptions.get("host");
        if (jythonHost == null) {
        	Map<String, String> providerConfigOptions = context.getConfigParams(
            		FloodlightProvider.class);
            jythonHost = providerConfigOptions.get("openflowhost");
        }
        if (jythonHost != null) {
        	log.debug("Jython host set to {}", jythonHost);
        }
        String port = configOptions.get("port");
        if (port != null) {
            jythonPort = Integer.parseInt(port);
        }
        log.debug("Jython port set to {}", jythonPort);
        JythonServer debug_server = new JythonServer(jythonHost, jythonPort, locals);
        debug_server.start();
    }
}
