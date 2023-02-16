package net.floodlightcontroller.core.internal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import net.floodlightcontroller.core.IShutdownListener;
import net.floodlightcontroller.core.IShutdownService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
public class ShutdownServiceImpl implements IFloodlightModule,
    IShutdownService {
    private static final Logger logger =
            LoggerFactory.getLogger(ShutdownServiceImpl.class);
    private static final int MAX_SHUTDOWN_WAIT_MS = 5_000;
    private final CopyOnWriteArrayList<IShutdownListener> shutdownListeners =
            new CopyOnWriteArrayList<>();
    @Override
    public Collection<Class<? extends IFloodlightService>> getModuleServices() {
        Collection<Class<? extends IFloodlightService>> l =
                new ArrayList<>();
        l.add(IShutdownService.class);
        return l;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
    getServiceImpls() {
        Map<Class<? extends IFloodlightService>, IFloodlightService> m =
                new HashMap<>();
        m.put(IShutdownService.class, this);
        return m;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
    getModuleDependencies() {
        return new ArrayList<>();
    }
    @Override
    public void init(FloodlightModuleContext context)
            throws FloodlightModuleException {
    }
    @Override
    public void startUp(FloodlightModuleContext context)
            throws FloodlightModuleException {
    }
    @Override
    public void registerShutdownListener(@Nonnull IShutdownListener listener) {
        if (listener == null) {
            throw new NullPointerException("listener must not be null");
        }
        shutdownListeners.add(listener);
    }
    @SuppressFBWarnings(value="DM_EXIT", justification="exit by design")
    @Override
    public void terminate(@Nullable final String reason, final int exitCode) {
        final String paddedReason;
        if (reason == null) {
            paddedReason = "";
        } else {
            paddedReason = " due to " + reason;
        }
        Thread shutdownForHungListeners =  new Thread(new Runnable() {
            @SuppressFBWarnings(value="DM_EXIT", justification="exit by design")
            @Override
            public void run() {
                try {
                    Thread.sleep(MAX_SHUTDOWN_WAIT_MS);
                } catch (InterruptedException e) {
                }
                System.exit(exitCode);
            }
        }, "ShutdownSafetyValve");
        shutdownForHungListeners.start();
        logger.info("Floodlight about to terminate. Calling shutdown listeners");
        for (IShutdownListener listener: shutdownListeners) {
            listener.floodlightIsShuttingDown();
        }
        if (exitCode == 0) {
        } else {
        }
        System.exit(exitCode);
    }
    @SuppressFBWarnings(value="DM_EXIT", justification="exit by design")
    @Override
    public void terminate(final String reason, final Throwable e, final int exitCode) {
        final String paddedReason;
        if (reason == null) {
            paddedReason = "";
        } else {
            paddedReason = " due to " + reason;
        }
        Thread shutdownForHungListeners =  new Thread(new Runnable() {
            @SuppressFBWarnings(value="DM_EXIT", justification="exit by design")
            @Override
            public void run() {
                try {
                    Thread.sleep(MAX_SHUTDOWN_WAIT_MS);
                } catch (InterruptedException e) {
                }
                logger.error("Floodlight is terminating{}", paddedReason, e);
                logger.error("ShutdownListeners failed to complete in time");
                System.exit(exitCode);
            }
        }, "ShutdownSafetyValve");
        shutdownForHungListeners.start();
        logger.info("Floodlight about to terminate. Calling shutdown listeners");
        for (IShutdownListener listener: shutdownListeners) {
            listener.floodlightIsShuttingDown();
        }
        logger.error("Floodlight is terminating abnormally{}", paddedReason, e);
        System.exit(exitCode);
    }
}
