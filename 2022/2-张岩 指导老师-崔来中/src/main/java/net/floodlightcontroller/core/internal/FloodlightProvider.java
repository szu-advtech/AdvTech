package net.floodlightcontroller.core.internal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.internal.Controller;
import net.floodlightcontroller.core.module.Run;
import org.sdnplatform.sync.ISyncService;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugevent.IDebugEventService;
import net.floodlightcontroller.perfmon.IPktInProcessingTimeService;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
public class FloodlightProvider implements IFloodlightModule {
    Controller controller;
    public FloodlightProvider() {
        controller = new Controller();
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        Collection<Class<? extends IFloodlightService>> services =
                new ArrayList<Class<? extends IFloodlightService>>(1);
        services.add(IFloodlightProviderService.class);
        return services;
    }
    @Override
    public Map<Class<? extends IFloodlightService>,
               IFloodlightService> getServiceImpls() {
        controller = new Controller();
        Map<Class<? extends IFloodlightService>,
            IFloodlightService> m =
                new HashMap<Class<? extends IFloodlightService>,
                            IFloodlightService>();
        m.put(IFloodlightProviderService.class, controller);
        return m;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
        Collection<Class<? extends IFloodlightService>> dependencies =
            new ArrayList<Class<? extends IFloodlightService>>(4);
        dependencies.add(IStorageSourceService.class);
        dependencies.add(IPktInProcessingTimeService.class);
        dependencies.add(IRestApiService.class);
        dependencies.add(IDebugCounterService.class);
        dependencies.add(IDebugEventService.class);
        dependencies.add(IOFSwitchService.class);
        dependencies.add(IThreadPoolService.class);
        dependencies.add(ISyncService.class);
        return dependencies;
    }
    @Override
    public void init(FloodlightModuleContext context) throws FloodlightModuleException {
       controller.setStorageSourceService(
           context.getServiceImpl(IStorageSourceService.class));
       controller.setPktInProcessingService(
           context.getServiceImpl(IPktInProcessingTimeService.class));
       controller.setDebugCounter(
           context.getServiceImpl(IDebugCounterService.class));
       controller.setDebugEvent(
           context.getServiceImpl(IDebugEventService.class));
       controller.setRestApiService(
           context.getServiceImpl(IRestApiService.class));
       controller.setThreadPoolService(
           context.getServiceImpl(IThreadPoolService.class));
       controller.setSyncService(
           context.getServiceImpl(ISyncService.class));
       controller.setSwitchService(
    	   context.getServiceImpl(IOFSwitchService.class));
       controller.init(context.getConfigParams(this));
    }
    @Override
    public void startUp(FloodlightModuleContext context)
            throws FloodlightModuleException {
        controller.startupComponents(context.getModuleLoader());
    }
    @Run(mainLoop=true)
    public void run() throws FloodlightModuleException {
        controller.run();
    }
}
