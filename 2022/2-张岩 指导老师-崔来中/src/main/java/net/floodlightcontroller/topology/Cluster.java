package net.floodlightcontroller.topology;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import net.floodlightcontroller.routing.Link;
import org.projectfloodlight.openflow.types.DatapathId;
public class Cluster {
    public Cluster() {
        id = DatapathId.NONE;
        links = new HashMap<DatapathId, Set<Link>>();
    }
    public DatapathId getId() {
        return id;
    }
    public void setId(DatapathId id) {
        this.id = id;
    }
    public Map<DatapathId, Set<Link>> getLinks() {
        return links;
    }
    public Set<DatapathId> getNodes() {
        return links.keySet();
    }
    void add(DatapathId n) {
        if (links.containsKey(n) == false) {
            links.put(n, new HashSet<Link>());
			if (id == DatapathId.NONE || n.getLong() < id.getLong()) 
				id = n ;
        }
    }
    void addLink(Link l) {
        add(l.getSrc());
        links.get(l.getSrc()).add(l);
        add(l.getDst());
        links.get(l.getDst()).add(l);
     }
    @Override 
    public int hashCode() {
        return (int) (id.getLong() + id.getLong() >>>32);
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Cluster other = (Cluster) obj;
        return (this.id.equals(other.id));
    }
    public String toString() {
        return "[Cluster id=" + id.toString() + ", " + links.keySet() + "]";
    }
}
