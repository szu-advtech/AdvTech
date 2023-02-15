package net.floodlightcontroller.core;
import java.net.SocketAddress;
import java.util.Date;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFAuxId;
import org.projectfloodlight.openflow.types.U64;
public interface IOFConnection extends IOFMessageWriter {
    Date getConnectedSince();
    DatapathId getDatapathId();
    OFAuxId getAuxId();
    SocketAddress getRemoteInetAddress();
    SocketAddress getLocalInetAddress();
    OFFactory getOFFactory();
    boolean isConnected();
	public U64 getLatency();
}
