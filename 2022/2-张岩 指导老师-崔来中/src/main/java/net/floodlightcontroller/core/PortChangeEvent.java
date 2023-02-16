package net.floodlightcontroller.core;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
public class PortChangeEvent {
    public final OFPortDesc port;
    public final PortChangeType type;
    public PortChangeEvent(OFPortDesc port,
                           PortChangeType type) {
        this.port = port;
        this.type = type;
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        PortChangeEvent other = (PortChangeEvent) obj;
        if (port == null) {
            if (other.port != null) return false;
        } else if (!port.equals(other.port)) return false;
        if (type != other.type) return false;
        return true;
    }
    @Override
    public String toString() {
        return "[" + type + " " + String.format("%s (%d)", port.getName(), port.getPortNo().getPortNumber()) + "]";
    }
}