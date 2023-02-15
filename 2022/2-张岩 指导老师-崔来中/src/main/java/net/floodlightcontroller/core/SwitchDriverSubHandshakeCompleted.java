package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.protocol.OFMessage;
public class SwitchDriverSubHandshakeCompleted
        extends SwitchDriverSubHandshakeException {
    private static final long serialVersionUID = -8817822245846375995L;
    public SwitchDriverSubHandshakeCompleted(OFMessage m) {
        super("Sub-Handshake is already complete but received message " +
              m.getType());
    }
}
