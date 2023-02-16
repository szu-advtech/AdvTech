package org.sdnplatform.sync.internal.rpc;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.LinkedTransferQueue;
import net.floodlightcontroller.core.util.SingletonTask;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import io.netty.bootstrap.Bootstrap;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.group.ChannelGroup;
import io.netty.channel.group.DefaultChannelGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.util.Timer;
import io.netty.util.concurrent.GlobalEventExecutor;
import org.sdnplatform.sync.internal.SyncManager;
import org.sdnplatform.sync.internal.config.Node;
import org.sdnplatform.sync.internal.util.Pair;
import org.sdnplatform.sync.thrift.SyncMessage;
import org.sdnplatform.sync.thrift.MessageType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class RPCService {
    protected static final Logger logger =
            LoggerFactory.getLogger(RPCService.class);
    protected SyncManager syncManager;
    protected IDebugCounterService debugCounter;
    private final ChannelGroup cg = new DefaultChannelGroup("Internal RPC", GlobalEventExecutor.INSTANCE);
    protected EventLoopGroup bossGroup;
    protected EventLoopGroup workerGroup;
    protected Bootstrap clientBootstrap;
    protected RPCChannelInitializer channelInitializer;
    protected HashMap<Short, NodeConnection> connections = 
            new HashMap<Short, NodeConnection>();
    protected AtomicInteger transactionId = new AtomicInteger();
    public static final int CONNECT_TIMEOUT = 500;
    protected boolean started = false;
    protected volatile boolean shutDown = false;
    protected SingletonTask reconnectTask;
    private final Timer timer;
    protected ConcurrentHashMap<Short, MessageWindow> messageWindows;
    protected static final EnumSet<MessageType> windowedTypes = 
            EnumSet.of(MessageType.SYNC_VALUE,
                       MessageType.SYNC_OFFER);
    protected ExecutorService syncExecutor;
    protected LinkedTransferQueue<NodeMessage> syncQueue = 
            new LinkedTransferQueue<NodeMessage>();
    protected static final int SYNC_MESSAGE_POOL = 2;
    protected static final int MAX_PENDING_MESSAGES = 500;
    public RPCService(SyncManager syncManager, 
                      IDebugCounterService debugCounter,
                      Timer timer) {
        super();
        this.syncManager = syncManager;
        this.debugCounter = debugCounter;
        this.timer = timer;
        messageWindows = new ConcurrentHashMap<Short, MessageWindow>();
    }
    public void run() {
        started = true;
        final ThreadGroup tg1 = new ThreadGroup("Sync Message Handlers");
        tg1.setMaxPriority(Thread.NORM_PRIORITY - 3);
        ThreadFactory f1 = new ThreadFactory() {
            AtomicInteger id = new AtomicInteger();
            @Override
            public Thread newThread(Runnable runnable) {
                return new Thread(tg1, runnable, 
                                  "SyncMessage-" + id.getAndIncrement());
            }
        };
        syncExecutor = Executors.newCachedThreadPool(f1);
        for (int i = 0; i < SYNC_MESSAGE_POOL; i++) {
            syncExecutor.execute(new SyncMessageWorker());
        }
        final ThreadGroup tg2 = new ThreadGroup("Sync I/O Threads");
        tg2.setMaxPriority(Thread.NORM_PRIORITY - 1);
        ThreadFactory f2 = new ThreadFactory() {
            @Override
            public Thread newThread(Runnable runnable) {
                return new Thread(tg2, runnable);
            }
        };
        bossGroup = new NioEventLoopGroup(0, f2);
        workerGroup = new NioEventLoopGroup(0, f2);
        channelInitializer = new RPCChannelInitializer(syncManager, this, timer);
        startServer(channelInitializer);
        startClients(channelInitializer);
    }
    public void shutdown() {
        shutDown = true;
        try {
            if (!cg.close().await(5, TimeUnit.SECONDS)) {
                logger.warn("Failed to cleanly shut down RPC server");
                return;
            }
            clientBootstrap = null;
            channelInitializer = null;
            if (bossGroup != null)
            	bossGroup.shutdownGracefully();
            bossGroup = null;
            if (workerGroup != null)
            	workerGroup.shutdownGracefully();
            workerGroup = null;
        } catch (InterruptedException e) {
            logger.warn("Interrupted while shutting down RPC server");
        }
        logger.debug("Internal floodlight RPC shut down");
    }
    public int getTransactionId() {
        return transactionId.getAndIncrement();
    }
    public boolean writeToNode(Short nodeId, SyncMessage bsm) 
            throws InterruptedException {
        if (nodeId == null) return false;
        NodeConnection nc = connections.get(nodeId);
        if (nc != null && nc.state == NodeConnectionState.CONNECTED) {
            waitForMessageWindow(bsm.getType(), nodeId, 0);
            nc.nodeChannel.writeAndFlush(bsm);
            return true;
        }
        return false;
    }
    public void disconnectNode(short nodeId) {
        synchronized (connections) {
            Short n = Short.valueOf(nodeId);
            MessageWindow mw = messageWindows.get(n);
            if (mw != null) {
                mw.lock.lock();
                mw.disconnected = true;
                try {
                    mw.full.signalAll();
                    messageWindows.remove(n);
                } finally {
                    mw.lock.unlock();
                }
            }
            NodeConnection nc = connections.get(nodeId);
            if (nc != null) {
                nc.nuke();
            }
            connections.remove(nodeId);
        }
    }
    public boolean isFullyConnected() {
        for (Node n : syncManager.getClusterConfig().getNodes()) {
            if (n.getNodeId() != syncManager.getLocalNodeId() &&
                !isConnected(n.getNodeId())) {
                if (logger.isTraceEnabled()) {
                    logger.trace("[{}->{}] missing connection",
                                 syncManager.getLocalNodeId(),
                                 n.getNodeId());
                }
                return false;
            }
        }
        return true;
    }
    public boolean isConnected(short nodeId) {
        NodeConnection nc = connections.get(nodeId);
        return (nc != null && nc.state == NodeConnectionState.CONNECTED);
    }
    public void messageAcked(MessageType type, Short nodeId) {
        if (nodeId == null) return;
        if (!windowedTypes.contains(type)) return;
        MessageWindow mw = messageWindows.get(nodeId);
        if (mw == null) return;
        int pending = mw.pending.decrementAndGet();
        if (pending < MAX_PENDING_MESSAGES) {
            mw.lock.lock();
            try {
                mw.full.signalAll();
            } finally {
                mw.lock.unlock();
            }
        }
    }
    private MessageWindow getMW(short nodeId) {
        if (!isConnected(nodeId)) return null;
        Short n = Short.valueOf(nodeId);
        MessageWindow mw = messageWindows.get(n);
        if (mw == null) {
            mw = new MessageWindow();
            MessageWindow old = messageWindows.putIfAbsent(n, mw);
            if (old != null) mw = old;
        }
        return mw;
    }
    private boolean waitForMessageWindow(MessageType type, short nodeId,
                                         long maxWait) 
            throws InterruptedException {
        if (!windowedTypes.contains(type)) return true;
        long start = System.nanoTime();
        MessageWindow mw = getMW(nodeId);
        if (!mw.disconnected && 
            mw.pending.get() >= MAX_PENDING_MESSAGES) {
            mw.lock.lock();
            try {
                while (!mw.disconnected && 
                       mw.pending.get() >= MAX_PENDING_MESSAGES) {
                    long now = System.nanoTime();
                    if (maxWait > 0 && 
                    mw.full.awaitNanos(now - start);
                }
            } finally {
                mw.lock.unlock();
            }
        }
        mw = getMW(nodeId);
        if (mw != null)
            mw.pending.getAndIncrement();
        return true;
    }
    protected void startServer(RPCChannelInitializer channelInitializer) {
        final ServerBootstrap bootstrap = new ServerBootstrap();
        bootstrap.group(bossGroup, workerGroup)
        .channel(NioServerSocketChannel.class)
        .option(ChannelOption.SO_REUSEADDR, true)
        .option(ChannelOption.SO_KEEPALIVE, true)
        .option(ChannelOption.TCP_NODELAY, true)
        .option(ChannelOption.SO_SNDBUF, SEND_BUFFER_SIZE)
        .option(ChannelOption.SO_RCVBUF, SEND_BUFFER_SIZE)
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, CONNECT_TIMEOUT)
        .childHandler(channelInitializer);
        int port = syncManager.getClusterConfig().getNode().getPort();
        InetSocketAddress sa;
        String listenAddress = 
                syncManager.getClusterConfig().getListenAddress();
        if (listenAddress != null)
            sa = new InetSocketAddress(listenAddress, port);
        else
            sa = new InetSocketAddress(port);
        ChannelFuture bindFuture = bootstrap.bind(sa);
        cg.add(bindFuture.channel());
        logger.info("Listening for internal floodlight RPC on {}", sa);
    }
    protected class ConnectCFListener implements ChannelFutureListener {
        protected Node node;
        public ConnectCFListener(Node node) {
            super();
            this.node = node;
        }
        @Override
        public void operationComplete(ChannelFuture cf) throws Exception {
            if (!cf.isSuccess()) {
                synchronized (connections) {
                    NodeConnection c = connections.remove(node.getNodeId());
                    if (c != null) c.nuke();
                    cf.channel().close();
                }
                String message = "[unknown error]";
                if (cf.isCancelled()) message = "Timed out on connect";
                if (cf.cause() != null) message = cf.cause().getMessage();
                logger.debug("[{}->{}] Could not connect to RPC " +
                             "node: {}", 
                             new Object[]{syncManager.getLocalNodeId(), 
                                          node.getNodeId(), 
                                          message});
            } else {
                logger.trace("[{}->{}] Channel future successful", 
                             syncManager.getLocalNodeId(), 
                             node.getNodeId());
            }
        }
    }
    protected void nodeConnected(short nodeId, Channel channel) {
        logger.debug("[{}->{}] Connection established",
                     syncManager.getLocalNodeId(),
                     nodeId);
        synchronized (connections) {
            NodeConnection c = connections.get(nodeId);
            if (c == null) {
                connections.put(nodeId, c = new NodeConnection());
            }
            c.nodeChannel = channel;
            c.state = NodeConnectionState.CONNECTED;
        }
    }
    protected void startClients(RPCChannelInitializer channelInitializer) {
        final Bootstrap bootstrap = new Bootstrap();
        bootstrap.group(workerGroup)
        .channel(NioSocketChannel.class)
        .option(ChannelOption.SO_REUSEADDR, true)
        .option(ChannelOption.SO_KEEPALIVE, true)
        .option(ChannelOption.TCP_NODELAY, true)
        .option(ChannelOption.SO_SNDBUF, SEND_BUFFER_SIZE)
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, CONNECT_TIMEOUT)
        .handler(channelInitializer);
        clientBootstrap = bootstrap;
        ScheduledExecutorService ses = 
                syncManager.getThreadPool().getScheduledExecutor();
        reconnectTask = new SingletonTask(ses, new ConnectTask());
        reconnectTask.reschedule(0, TimeUnit.SECONDS);
    }
    protected void doNodeConnect(Node n) {
        if (!shutDown && n.getNodeId() < syncManager.getLocalNodeId()) {
            Short nodeId = n.getNodeId();
            synchronized (connections) {
                NodeConnection c = connections.get(n.getNodeId());
                if (c == null) {
                    connections.put(nodeId, c = new NodeConnection());
                }
                if (logger.isTraceEnabled()) {
                    logger.trace("[{}->{}] Connection state: {}", 
                                 new Object[]{syncManager.getLocalNodeId(),
                                              nodeId, c.state});
                }
                if (c.state.equals(NodeConnectionState.NONE)) {
                    if (logger.isDebugEnabled()) {
                        logger.debug("[{}->{}] Attempting connection {} {}", 
                                     new Object[]{syncManager.getLocalNodeId(),
                                                  nodeId, 
                                                  n.getHostname(), 
                                                  n.getPort()});
                    }
                    SocketAddress sa =
                            new InetSocketAddress(n.getHostname(), n.getPort());
                    c.pendingFuture = clientBootstrap.connect(sa);
                    c.pendingFuture.addListener(new ConnectCFListener(n));
                    c.state = NodeConnectionState.PENDING;
                }
            }
        }
    }
    protected void startClientConnections() {
        for (Node n : syncManager.getClusterConfig().getNodes()) {
            doNodeConnect(n);
        }        
    }
    protected ChannelGroup getChannelGroup() {
    	return cg;
    }
    protected class ConnectTask implements Runnable {
        @Override
        public void run() {
            try {
                if (!shutDown)
                    startClientConnections();
            } catch (Exception e) {
                logger.error("Error in reconnect task", e);
            }
            if (!shutDown) {
                reconnectTask.reschedule(500, TimeUnit.MILLISECONDS);
            }
        }
    }
    protected enum NodeConnectionState {
        NONE,
        PENDING,
        CONNECTED
    }
    protected static class NodeConnection {
        volatile NodeConnectionState state = NodeConnectionState.NONE;        
        protected ChannelFuture pendingFuture;
        protected Channel nodeChannel;
        protected void nuke() {
            state = NodeConnectionState.NONE;
            if (pendingFuture != null) pendingFuture.cancel(false);
            if (nodeChannel != null) nodeChannel.close();
            pendingFuture = null;
            nodeChannel = null;
        }
    }
    protected static class MessageWindow {
        AtomicInteger pending = new AtomicInteger();
        volatile boolean disconnected = false;
        Lock lock = new ReentrantLock();
        Condition full = lock.newCondition();
    }
    protected static class NodeMessage extends Pair<Short,SyncMessage> {
        private static final long serialVersionUID = -3443080461324647922L;
        public NodeMessage(Short first, SyncMessage second) {
            super(first, second);
        }
    }
    protected class SyncMessageWorker implements Runnable {
        @Override
        public void run() {
            while (true) {
                try {
                    NodeMessage m = syncQueue.take();
                    writeToNode(m.getFirst(), m.getSecond());
                } catch (Exception e) {
                    logger.error("Error while dispatching message", e);
                }
            }
        }
    }
}