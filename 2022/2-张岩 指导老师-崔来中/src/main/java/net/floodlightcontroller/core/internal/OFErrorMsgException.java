package net.floodlightcontroller.core.internal;
import org.projectfloodlight.openflow.protocol.OFErrorMsg;
import org.projectfloodlight.openflow.protocol.OFRequest;
public class OFErrorMsgException extends Exception {
    private static final long serialVersionUID = 1L;
    private final OFErrorMsg errorMessage;
    public OFErrorMsgException(final OFErrorMsg errorMessage) {
        super("OF error received: " + errorMessage.toString());
        this.errorMessage = errorMessage;
    }
    public OFErrorMsg getErrorMessage() {
        return errorMessage;
    }
}
