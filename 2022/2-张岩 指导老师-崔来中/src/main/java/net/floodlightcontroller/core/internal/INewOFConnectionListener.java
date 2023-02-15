package net.floodlightcontroller.core.internal;
import net.floodlightcontroller.core.IOFConnectionBackend;
import org.projectfloodlight.openflow.protocol.OFFeaturesReply;
public interface INewOFConnectionListener {
    void connectionOpened(IOFConnectionBackend connection,
                          OFFeaturesReply featuresReply);
}
