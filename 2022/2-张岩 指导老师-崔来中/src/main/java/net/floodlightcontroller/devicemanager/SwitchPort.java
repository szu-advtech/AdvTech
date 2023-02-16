package net.floodlightcontroller.devicemanager;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.core.web.serializers.DPIDSerializer;
import net.floodlightcontroller.core.web.serializers.OFPortSerializer;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
public class SwitchPort {
    @JsonSerialize(using=ToStringSerializer.class)
    public enum ErrorStatus {
        DUPLICATE_DEVICE("duplicate-device");
        private String value;
        ErrorStatus(String v) {
            value = v;
        }
        @Override
        public String toString() {
            return value;
        }
        public static ErrorStatus fromString(String str) {
            for (ErrorStatus m : ErrorStatus.values()) {
                if (m.value.equals(str)) {
                    return m;
                }
            }
            return null;
        }
    }
    private final DatapathId switchDPID;
    private final OFPort port;
    private final ErrorStatus errorStatus;
    public SwitchPort(DatapathId switchDPID, OFPort port, ErrorStatus errorStatus) {
        super();
        this.switchDPID = switchDPID;
        this.port = port;
        this.errorStatus = errorStatus;
    }
    public SwitchPort(DatapathId switchDPID, OFPort port) {
        super();
        this.switchDPID = switchDPID;
        this.port = port;
        this.errorStatus = null;
    }
    @JsonSerialize(using=DPIDSerializer.class)
    public DatapathId getSwitchDPID() {
        return switchDPID;
    }
    @JsonSerialize(using=OFPortSerializer.class)
    public OFPort getPort() {
        return port;
    }
    public ErrorStatus getErrorStatus() {
        return errorStatus;
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
                        + ((errorStatus == null)
                                ? 0
                                : errorStatus.hashCode());
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        SwitchPort other = (SwitchPort) obj;
        if (errorStatus != other.errorStatus) return false;
        if (!port.equals(other.port)) return false;
        if (!switchDPID.equals(other.switchDPID)) return false;
        return true;
    }
    @Override
    public String toString() {
        return "SwitchPort [switchDPID=" + switchDPID.toString() +
               ", port=" + port + ", errorStatus=" + errorStatus + "]";
    }
}
