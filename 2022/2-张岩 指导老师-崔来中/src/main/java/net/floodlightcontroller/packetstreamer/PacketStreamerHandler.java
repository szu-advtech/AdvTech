package net.floodlightcontroller.packetstreamer;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Map;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class PacketStreamerHandler implements PacketStreamer.Iface {
    protected class SessionQueue {
        protected BlockingQueue<ByteBuffer> pQueue;
        public SessionQueue() {
            this.pQueue = new LinkedBlockingQueue<ByteBuffer>();
        }
        public BlockingQueue<ByteBuffer> getQueue() {
            return this.pQueue;
        }
    }
    protected static Logger log = 
            LoggerFactory.getLogger(PacketStreamerServer.class);
    protected Map<String, SessionQueue> msgQueues;
    public PacketStreamerHandler() {
        this.msgQueues = new ConcurrentHashMap<String, SessionQueue>();
    }
    @Override
    public List<ByteBuffer> getPackets(String sessionid)
            throws org.apache.thrift.TException {
        List<ByteBuffer> packets = new ArrayList<ByteBuffer>();
        int count = 0;
        while (!msgQueues.containsKey(sessionid) && count++ < 100) {
            log.debug("Queue for session {} doesn't exist yet.", sessionid);
            try {
            } catch (InterruptedException e) {
                log.error("Interrupted while waiting for session start");
            }
        }
        if (count < 100) {
	        SessionQueue pQueue = msgQueues.get(sessionid);
	        BlockingQueue<ByteBuffer> queue = pQueue.getQueue();
	        try {
	            packets.add(queue.take());
	            queue.drainTo(packets);
	        } catch (InterruptedException e) {
	            log.error("Interrupted while waiting for packets");
	        }
        }
        return packets;
    }
    @Override
    public int pushMessageSync(Message msg)
            throws org.apache.thrift.TException {
        if (msg == null) {
            log.error("Could not push empty message");
            return 0;
        }
        List<String> sessionids = msg.getSessionIDs();
        for (String sid : sessionids) {
            SessionQueue pQueue = null;
            if (!msgQueues.containsKey(sid)) {
                pQueue = new SessionQueue();
                msgQueues.put(sid, pQueue);
            } else {
                pQueue = msgQueues.get(sid);
            }
            log.debug("pushMessageSync: SessionId: " + sid + 
                      " Receive a message, " + msg.toString() + "\n");
            ByteBuffer bb = ByteBuffer.wrap(msg.getPacket().getData());
            BlockingQueue<ByteBuffer> queue = pQueue.getQueue();
            if (queue != null) {
                if (!queue.offer(bb)) {
                    log.error("Failed to queue message for session: " + sid);
                } else {
                    log.debug("insert a message to session: " + sid);
                }
            } else {
                log.error("queue for session {} is null", sid);
            }
        }
        return 1;
    }
    @Override
    public void pushMessageAsync(Message msg)
            throws org.apache.thrift.TException {
        pushMessageSync(msg);
        return;
    }
    @Override
    public void terminateSession(String sessionid)
            throws org.apache.thrift.TException {
        if (!msgQueues.containsKey(sessionid)) {
            return;
        }
        SessionQueue pQueue = msgQueues.get(sessionid);
        log.debug("terminateSession: SessionId: " + sessionid + "\n");
        String data = "FilterTimeout";
        ByteBuffer bb = ByteBuffer.wrap(data.getBytes());
        BlockingQueue<ByteBuffer> queue = pQueue.getQueue();
        if (queue != null) {
            if (!queue.offer(bb)) {
                log.error("Failed to queue message for session: " + sessionid);
            }
            msgQueues.remove(sessionid);
        } else {
            log.error("queue for session {} is null", sessionid);
        }
    }
}
