package net.floodlightcontroller.core;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IShutdownService extends IFloodlightService {
    public void terminate(@Nullable String reason, int exitCode);
    public void terminate(@Nullable String reason,
                          @Nonnull Throwable e, int exitCode);
    public void registerShutdownListener(@Nonnull IShutdownListener listener);
}
