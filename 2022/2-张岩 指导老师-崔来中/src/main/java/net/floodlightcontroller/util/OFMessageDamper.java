package net.floodlightcontroller.util;
import java.io.IOException;
import java.util.EnumSet;
import java.util.Set;
import net.floodlightcontroller.core.IOFSwitch;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFType;
public class OFMessageDamper {
    protected static class DamperEntry {
        OFMessage msg;
        IOFSwitch sw;
        public DamperEntry(OFMessage msg, IOFSwitch sw) {
            super();
            this.msg = msg;
            this.sw = sw;
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
            DamperEntry other = (DamperEntry) obj;
            if (msg == null) {
                if (other.msg != null) return false;
            } else if (!msg.equals(other.msg)) return false;
            if (sw == null) {
                if (other.sw != null) return false;
            } else if (!sw.equals(other.sw)) return false;
            return true;
        }
    }
    TimedCache<DamperEntry> cache;
    EnumSet<OFType> msgTypesToCache;
    public OFMessageDamper(int capacity, 
                           Set<OFType> typesToDampen,  
                           int timeout) {
        cache = new TimedCache<DamperEntry>(capacity, timeout);
        msgTypesToCache = EnumSet.copyOf(typesToDampen);
    }        
    public boolean write(IOFSwitch sw, OFMessage msg) throws IOException {
        if (!msgTypesToCache.contains(msg.getType())) {
            sw.write(msg);
            return true;
        }
        DamperEntry entry = new DamperEntry(msg, sw);
        if (cache.update(entry)) {
            return false; 
        } else {
            sw.write(msg);
            return true;
        }
    }
}
