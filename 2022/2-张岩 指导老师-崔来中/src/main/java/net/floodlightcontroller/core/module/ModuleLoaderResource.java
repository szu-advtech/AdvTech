package net.floodlightcontroller.core.module;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class ModuleLoaderResource extends ServerResource {
    protected static Logger log = 
            LoggerFactory.getLogger(ModuleLoaderResource.class);
    @Get("json")
    public Map<String, Object> retrieve() {
    	return retrieveInternal(false);
    }
    public Map<String, Object> retrieveInternal(boolean loadedOnly) {    
        Map<String, Object> model = new HashMap<String, Object>();
        FloodlightModuleLoader floodlightModuleLoader =
                (FloodlightModuleLoader) getContext().getAttributes().
                get(FloodlightModuleLoader.class.getCanonicalName());
        Set<String> loadedModules = new HashSet<String>();
        for (Object val : getContext().getAttributes().values()) {
        	if ((val instanceof IFloodlightModule) || (val instanceof IFloodlightService)) {
        		String serviceImpl = val.getClass().getCanonicalName();
        		loadedModules.add(serviceImpl);
        	}
        }
        for (String moduleName : floodlightModuleLoader.getModuleNameMap().keySet() ) {
        	Map<String,Object> moduleInfo = new HashMap<String, Object>();
        	IFloodlightModule module = 
        			floodlightModuleLoader.getModuleNameMap().get(moduleName);
        	Collection<Class<? extends IFloodlightService>> deps = 
        			module.getModuleDependencies();
        	if ( deps == null)
            	deps = new HashSet<Class<? extends IFloodlightService>>();
        	Map<String,Object> depsMap = new HashMap<String, Object> ();
        	for (Class<? extends IFloodlightService> service : deps) {
        		Object serviceImpl = getContext().getAttributes().get(service.getCanonicalName());
        		if (serviceImpl != null)
        			depsMap.put(service.getCanonicalName(), serviceImpl.getClass().getCanonicalName());
        		else
        			depsMap.put(service.getCanonicalName(), "<unresolved>");
        	}
            moduleInfo.put("depends", depsMap);
            Collection<Class<? extends IFloodlightService>> provides = 
            		module.getModuleServices();
        	if ( provides == null)
            	provides = new HashSet<Class<? extends IFloodlightService>>();
        	Map<String,Object> providesMap = new HashMap<String,Object>();
        	for (Class<? extends IFloodlightService> service : provides) {
        		providesMap.put(service.getCanonicalName(), module.getServiceImpls().get(service).getClass().getCanonicalName());
        	}
        	moduleInfo.put("provides", providesMap);            		
        	if (loadedModules.contains(module.getClass().getCanonicalName())) {
        		moduleInfo.put("loaded", true);  			
        	} else {
        		for (Class<? extends IFloodlightService> service : provides) {
        			String modString = module.getServiceImpls().get(service).getClass().getCanonicalName();
        			if (loadedModules.contains(modString))
                		moduleInfo.put("loaded", true);
        		}
        	}
        	if ((Boolean)moduleInfo.get("loaded")|| !loadedOnly )
        		model.put(moduleName, moduleInfo);
        }            
        return model;
    }
}