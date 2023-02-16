package org.sdnplatform.sync.internal.rpc;
import java.util.concurrent.TimeUnit;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.util.Timeout;
import io.netty.util.Timer;
import io.netty.util.TimerTask;
import org.sdnplatform.sync.error.HandshakeTimeoutException;
public class HandshakeTimeoutHandler extends ChannelInboundHandlerAdapter {
    static final HandshakeTimeoutException EXCEPTION = 
            new HandshakeTimeoutException();
    final RPCChannelHandler handler;
    final Timer timer;
    final long timeoutNanos;
    volatile Timeout timeout;
    public HandshakeTimeoutHandler(RPCChannelHandler handler,
                                   Timer timer,
                                   long timeoutSeconds) {
        super();
        this.handler = handler;
        this.timer = timer;
        this.timeoutNanos = TimeUnit.SECONDS.toNanos(timeoutSeconds);
    }
    @Override
    public void channelActive(ChannelHandlerContext ctx)
            throws Exception {
        if (timeoutNanos > 0) {
            timeout = timer.newTimeout(new HandshakeTimeoutTask(ctx), 
                                       timeoutNanos, TimeUnit.NANOSECONDS);
        }
        ctx.fireChannelActive();
    }
    @Override
    public void channelInactive(ChannelHandlerContext ctx)
            throws Exception {
        if (timeout != null) {
            timeout.cancel();
            timeout = null;
        }
        ctx.fireChannelInactive();
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
            if (!handler.isClientConnection && 
                ((handler.remoteNode == null ||
                 !handler.rpcService.isConnected(handler.remoteNode.
                                                 getNodeId()))))
                ctx.fireExceptionCaught(EXCEPTION);
        }
    }
}
