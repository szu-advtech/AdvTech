package net.floodlightcontroller.devicemanager.internal;
import java.util.Date;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
public class AttachmentPoint {
    DatapathId  sw;
    OFPort port;
    Date  activeSince;
    Date  lastSeen;
    public AttachmentPoint(DatapathId sw, OFPort port, Date activeSince, Date lastSeen) {
        this.sw = sw;
        this.port = port;
        this.activeSince = activeSince;
        this.lastSeen = lastSeen;
    }
    public AttachmentPoint(DatapathId sw, OFPort port, Date lastSeen) {
        this.sw = sw;
        this.port = port;
        this.lastSeen = lastSeen;
        this.activeSince = lastSeen;
    }
    public AttachmentPoint(AttachmentPoint ap) {
        this.sw = ap.getSw();
        this.port = ap.port;
        this.activeSince = ap.activeSince;
        this.lastSeen = ap.lastSeen;
    }
    public DatapathId getSw() {
        return sw;
    }
    public void setSw(DatapathId sw) {
        this.sw = sw;
    }
    public OFPort getPort() {
        return port;
    }
    public void setPort(OFPort port) {
        this.port = port;
    }
    public Date getActiveSince() {
        return activeSince;
    }
    public void setActiveSince(Date activeSince) {
        this.activeSince = activeSince;
    }
    public Date getLastSeen() {
        return lastSeen;
    }
    public void setLastSeen(Date lastSeen) {
        if (this.lastSeen.getTime() + INACTIVITY_INTERVAL < lastSeen.getTime())
            this.activeSince = lastSeen;
        if (this.lastSeen.before(lastSeen))
            this.lastSeen = lastSeen;
    }
    @Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		return result;
	}
    @Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		AttachmentPoint other = (AttachmentPoint) obj;
		if (port == null) {
			if (other.port != null)
				return false;
		} else if (!port.equals(other.port))
			return false;
		if (sw == null) {
			if (other.sw != null)
				return false;
		} else if (!sw.equals(other.sw))
			return false;
		return true;
	}
    @Override
    public String toString() {
        return "AttachmentPoint [sw=" + sw + ", port=" + port
               + ", activeSince=" + activeSince + ", lastSeen=" + lastSeen.toString()
               + "]";
    }
}