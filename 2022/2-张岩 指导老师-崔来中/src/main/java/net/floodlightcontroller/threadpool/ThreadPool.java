package net.floodlightcontroller.threadpool;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
public class ThreadPool implements IThreadPoolService, IFloodlightModule {
    protected ScheduledExecutorService executor = null;
    @Override
    public ScheduledExecutorService getScheduledExecutor() {
        return executor;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        Collection<Class<? extends IFloodlightService>> l = 
                new ArrayList<Class<? extends IFloodlightService>>();
        l.add(IThreadPoolService.class);
        return l;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
            getServiceImpls() {
        Map<Class<? extends IFloodlightService>,
            IFloodlightService> m = 
                new HashMap<Class<? extends IFloodlightService>,
                    IFloodlightService>();
        m.put(IThreadPoolService.class, this);
        return m;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
            getModuleDependencies() {
        return null;
    }
    @Override
    public void init(FloodlightModuleContext context)
                                 throws FloodlightModuleException {
        final ThreadGroup tg = new ThreadGroup("Scheduled Task Threads");
        ThreadFactory f = new ThreadFactory() {
            AtomicInteger id = new AtomicInteger();
            @Override
            public Thread newThread(Runnable runnable) {
                return new Thread(tg, runnable, 
                                  "Scheduled-" + id.getAndIncrement());
            }
        };
        executor = Executors.newScheduledThreadPool(5, f);
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
    }
}
