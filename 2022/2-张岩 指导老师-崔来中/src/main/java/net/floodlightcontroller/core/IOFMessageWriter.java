package net.floodlightcontroller.core;
import java.util.Collection;
import java.util.List;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFRequest;
import org.projectfloodlight.openflow.protocol.OFStatsReply;
import org.projectfloodlight.openflow.protocol.OFStatsRequest;
import com.google.common.util.concurrent.ListenableFuture;
public interface IOFMessageWriter{
    boolean write(OFMessage m);
    Collection<OFMessage> write(Iterable<OFMessage> msgList);
    <R extends OFMessage> ListenableFuture<R> writeRequest(OFRequest<R> request);
    <REPLY extends OFStatsReply> ListenableFuture<List<REPLY>> writeStatsRequest(
            OFStatsRequest<REPLY> request);
}
