package net.floodlightcontroller.routing;
import java.util.ArrayList;
import java.util.List;
import org.projectfloodlight.openflow.types.DatapathId;
import net.floodlightcontroller.topology.NodePortTuple;
public class Route implements Comparable<Route> {
    protected RouteId id;
    protected List<NodePortTuple> switchPorts;
    protected int routeCount;
    public Route(RouteId id, List<NodePortTuple> switchPorts) {
        super();
        this.id = id;
        this.switchPorts = switchPorts;
    }
    public Route(DatapathId src, DatapathId dst) {
        super();
        this.id = new RouteId(src, dst);
        this.switchPorts = new ArrayList<NodePortTuple>();
        this.routeCount = 0;
    }
    public RouteId getId() {
        return id;
    }
    public void setId(RouteId id) {
        this.id = id;
    }
    public List<NodePortTuple> getPath() {
        return switchPorts;
    }
    public void setPath(List<NodePortTuple> switchPorts) {
        this.switchPorts = switchPorts;
    }
    public void setRouteCount(int routeCount) {
        this.routeCount = routeCount;
    }
    public int getRouteCount() {
        return routeCount;
    }
    @Override
    public int hashCode() {
        final int prime = 5791;
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
        Route other = (Route) obj;
        if (id == null) {
            if (other.id != null)
                return false;
        } else if (!id.equals(other.id))
            return false;
        if (switchPorts == null) {
            if (other.switchPorts != null)
                return false;
        } else if (!switchPorts.equals(other.switchPorts))
            return false;
        return true;
    }
    @Override
    public String toString() {
        return "Route [id=" + id + ", switchPorts=" + switchPorts + "]";
    }
    @Override
    public int compareTo(Route o) {
        return ((Integer)switchPorts.size()).compareTo(o.switchPorts.size());
    }
}
