package net.floodlightcontroller.core.types;
import org.projectfloodlight.openflow.protocol.OFMessage;
import net.floodlightcontroller.core.IOFSwitch;
public class SwitchMessagePair {
    private final IOFSwitch sw;
    private final OFMessage msg;
    public SwitchMessagePair(IOFSwitch sw, OFMessage msg) {
        this.sw = sw;
        this.msg = msg;
    }
    public IOFSwitch getSwitch() {
        return this.sw;
    }
    public OFMessage getMessage() {
        return this.msg;
    }
}
