package net.floodlightcontroller.core.internal;
import java.util.concurrent.TimeUnit;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.util.Timeout;
import io.netty.util.Timer;
import io.netty.util.TimerTask;
public class HandshakeTimeoutHandler extends ChannelInboundHandlerAdapter {
    static final HandshakeTimeoutException EXCEPTION = new HandshakeTimeoutException();
    final OFChannelHandler handshakeHandler;
    final Timer timer;
    final long timeoutNanos;
    volatile Timeout timeout;
    public HandshakeTimeoutHandler(OFChannelHandler handshakeHandler,
                                   Timer timer,
                                   long timeoutSeconds) {
        super();
        this.handshakeHandler = handshakeHandler;
        this.timer = timer;
        this.timeoutNanos = TimeUnit.SECONDS.toNanos(timeoutSeconds);
    }
    @Override
    public void channelActive(ChannelHandlerContext ctx) throws Exception {
        if (timeoutNanos > 0) {
            timeout = timer.newTimeout(new HandshakeTimeoutTask(ctx), timeoutNanos, TimeUnit.NANOSECONDS);
        }
        super.channelActive(ctx);
    }
    @Override
    public void channelInactive(ChannelHandlerContext ctx) throws Exception {
        if (timeout != null) {
            timeout.cancel();
            timeout = null;
        }
        super.channelInactive(ctx);
    }
    private final class HandshakeTimeoutTask implements TimerTask {
        private final ChannelHandlerContext ctx;
        HandshakeTimeoutTask(ChannelHandlerContext ctx) {
            this.ctx = ctx;
        }
        @Override
        public void run(Timeout timeout) throws Exception {
            if (timeout.isCancelled()) {
                return;
            }
            if (!ctx.channel().isOpen()) {
                return;
            }
            if (!handshakeHandler.isSwitchHandshakeComplete())
                ctx.fireExceptionCaught(EXCEPTION);
        }
    }
}