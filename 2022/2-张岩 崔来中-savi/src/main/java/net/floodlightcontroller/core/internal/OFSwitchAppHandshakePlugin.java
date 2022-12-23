package net.floodlightcontroller.core.internal;
import java.util.concurrent.TimeUnit;
import io.netty.util.Timeout;
import io.netty.util.Timer;
import io.netty.util.TimerTask;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.OFSwitchHandshakeHandler.WaitAppHandshakeState;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.base.Preconditions;
public abstract class OFSwitchAppHandshakePlugin {
    private static final Logger log = LoggerFactory.getLogger(OFSwitchAppHandshakePlugin.class);
    private WaitAppHandshakeState state;
    @edu.umd.cs.findbugs.annotations.SuppressWarnings("URF_UNREAD_PUBLIC_OR_PROTECTED_FIELD")
    private IOFSwitch sw;
    private volatile Timeout timeout;
    private final PluginResult defaultResult;
    private final int timeoutS;
    protected OFSwitchAppHandshakePlugin(PluginResult defaultResult, int timeoutS){
        Preconditions.checkNotNull(defaultResult, "defaultResult");
        Preconditions.checkNotNull(timeoutS, "timeoutS");
        this.defaultResult = defaultResult;
        this.timeoutS = timeoutS;
    }
    protected abstract void processOFMessage(OFMessage m);
    protected abstract void enterPlugin();
    protected IOFSwitch getSwitch() {
        return this.sw;
    }
    final void init(WaitAppHandshakeState state, IOFSwitch sw, Timer timer) {
        this.state = state;
        this.sw = sw;
        this.timeout = timer.newTimeout(new PluginTimeoutTask(), timeoutS, TimeUnit.SECONDS);
    }
    protected final void exitPlugin(PluginResult result) {
        timeout.cancel();
        state.exitPlugin(result);
    }
    private final class PluginTimeoutTask implements TimerTask {
        @Override
        public void run(Timeout timeout) throws Exception {
            if (!timeout.isCancelled()) {
                log.warn("App handshake plugin for {} timed out. Returning result {}.",
                         sw, defaultResult);
                exitPlugin(defaultResult);
            }
        }
    }
    public enum PluginResultType {
        CONTINUE(),
        DISCONNECT(),
        QUARANTINE();
    }
}
