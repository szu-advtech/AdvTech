package net.floodlightcontroller.threadpool;
import java.util.concurrent.ScheduledExecutorService;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IThreadPoolService extends IFloodlightService {
    public ScheduledExecutorService getScheduledExecutor();
}
