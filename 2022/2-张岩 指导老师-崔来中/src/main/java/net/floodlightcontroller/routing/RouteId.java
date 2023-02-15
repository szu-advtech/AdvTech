package net.floodlightcontroller.routing;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.U64;
public class RouteId implements Cloneable, Comparable<RouteId> {
    protected DatapathId src;
    protected DatapathId dst;
    protected U64 cookie;
    public RouteId(DatapathId src, DatapathId dst) {
        super();
        this.src = src;
        this.dst = dst;
        this.cookie = U64.of(0);
    }
    public RouteId(DatapathId src, DatapathId dst, U64 cookie) {
        super();
        this.src = src;
        this.dst = dst;
        this.cookie = cookie;
    }
    public DatapathId getSrc() {
        return src;
    }
    public void setSrc(DatapathId src) {
        this.src = src;
    }
    public DatapathId getDst() {
        return dst;
    }
    public void setDst(DatapathId dst) {
        this.dst = dst;
    }
    public U64 getCookie() {
        return cookie;
    }
    public void setCookie(int cookie) {
        this.cookie = U64.of(cookie);
    }
    @Override
    public int hashCode() {
        final int prime = 2417;
        Long result = new Long(1);
        return result.hashCode(); 
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        RouteId other = (RouteId) obj;
        if (dst == null) {
            if (other.dst != null)
                return false;
        } else if (!dst.equals(other.dst))
            return false;
        if (src == null) {
            if (other.src != null)
                return false;
        } else if (!src.equals(other.src))
            return false;
        return true;
    }
    @Override
    public String toString() {
        return "RouteId [src=" + this.src.toString() + " dst="
                + this.dst.toString() + "]";
    }
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
    @Override
    public int compareTo(RouteId o) {
        int result = src.compareTo(o.getSrc());
        if (result != 0)
            return result;
        return dst.compareTo(o.getDst());
    }
}
