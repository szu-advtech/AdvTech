package net.floodlightcontroller.routing;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
public class Link implements Comparable<Link> {
    @JsonProperty("src-switch")
    private DatapathId src;
    @JsonProperty("src-port")
    private OFPort srcPort;
    @JsonProperty("dst-switch")
    private DatapathId dst;
    @JsonProperty("dst-port")
    private OFPort dstPort;
    @JsonProperty("latency") 
    public Link(DatapathId srcId, OFPort srcPort, DatapathId dstId, OFPort dstPort, U64 latency) {
        this.src = srcId;
        this.srcPort = srcPort;
        this.dst = dstId;
        this.dstPort = dstPort;
        this.latency = latency;
    }
    public Link() {
        super();
    }
    public DatapathId getSrc() {
        return src;
    }
    public OFPort getSrcPort() {
        return srcPort;
    }
    public DatapathId getDst() {
        return dst;
    }
    public OFPort getDstPort() {
        return dstPort;
    }
    public U64 getLatency() {
    	return latency;
    }
    public void setSrc(DatapathId src) {
        this.src = src;
    }
    public void setSrcPort(OFPort srcPort) {
        this.srcPort = srcPort;
    }
    public void setDst(DatapathId dst) {
        this.dst = dst;
    }
    public void setDstPort(OFPort dstPort) {
        this.dstPort = dstPort;
    }
    public void setLatency(U64 latency) {
    	this.latency = latency;
    }
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Link other = (Link) obj;
        if (!dst.equals(other.dst))
            return false;
        if (!dstPort.equals(other.dstPort))
            return false;
        if (!src.equals(other.src))
            return false;
        if (!srcPort.equals(other.srcPort))
            return false;
    }
    @Override
    public String toString() {
        return "Link [src=" + this.src.toString() 
                + " outPort="
                + srcPort.toString()
                + ", dst=" + this.dst.toString()
                + ", inPort="
                + dstPort.toString()
                + ", latency="
                + String.valueOf(latency.getValue())
                + "]";
    }
    public String toKeyString() {
    	return (this.src.toString() + "|" +
    			this.srcPort.toString() + "|" +
    			this.dst.toString() + "|" +
    		    this.dstPort.toString());
    }
    @Override
    public int compareTo(Link a) {
        int srcComp = this.getSrc().compareTo(a.getSrc());
        if (srcComp != 0)
            return srcComp;
        int srcPortComp = this.getSrcPort().compareTo(a.getSrcPort());
        if (srcPortComp != 0)
            return srcPortComp;
        int dstComp = this.getDst().compareTo(a.getDst());
        if (dstComp != 0)
            return dstComp;
        return this.getDstPort().compareTo(a.getDstPort());
    }
}