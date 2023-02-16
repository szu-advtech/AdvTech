package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.types.U64;
import net.floodlightcontroller.core.internal.IOFConnectionListener;
public interface IOFConnectionBackend extends IOFConnection {
    void disconnect();
    void cancelAllPendingRequests();
    boolean isWritable();
    void setListener(IOFConnectionListener listener);
    public void updateLatency(U64 latency);
}
