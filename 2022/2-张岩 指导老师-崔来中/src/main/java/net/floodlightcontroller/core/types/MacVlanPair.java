package net.floodlightcontroller.core.types;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.VlanVid;
public class MacVlanPair {
    public MacAddress mac;
    public VlanVid vlan;
    public MacVlanPair(MacAddress mac2, VlanVid vlan2) {
        this.mac = mac2;
        this.vlan = vlan2;
    }
    public MacAddress getMac() {
        return mac;
    }
    public VlanVid getVlan() {
        return vlan;
    }
    public boolean equals(Object o) {
        return (o instanceof MacVlanPair) && (mac.equals(((MacVlanPair) o).mac))
            && (vlan.equals(((MacVlanPair) o).vlan));
    }
    public int hashCode() {
        return mac.hashCode() ^ vlan.hashCode();
    }
    public String toString() {
        return "(" + mac.toString() + ", " + vlan.toString() + ")";
    }
}
