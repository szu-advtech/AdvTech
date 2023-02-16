package net.floodlightcontroller.routing;
import java.util.HashMap;
import net.floodlightcontroller.routing.Link;
import org.projectfloodlight.openflow.types.DatapathId;
public class BroadcastTree {
    protected HashMap<DatapathId, Link> links;
    protected HashMap<DatapathId, Integer> costs;
    public BroadcastTree() {
        links = new HashMap<DatapathId, Link>();
        costs = new HashMap<DatapathId, Integer>();
    }
    public BroadcastTree(HashMap<DatapathId, Link> links, HashMap<DatapathId, Integer> costs) {
        this.links = links;
        this.costs = costs;
    }
    public Link getTreeLink(DatapathId node) {
        return links.get(node);
    }
    public int getCost(DatapathId node) {
        if (costs.get(node) == null) return -1;
        return (costs.get(node));
    }
    public HashMap<DatapathId, Link> getLinks() {
        return links;
    }
    public void addTreeLink(DatapathId myNode, Link link) {
        links.put(myNode, link);
    }
    public String toString() {
        StringBuffer sb = new StringBuffer();
        for(DatapathId n: links.keySet()) {
            sb.append("[" + n.toString() + ": cost=" + costs.get(n) + ", " + links.get(n) + "]");
        }
        return sb.toString();
    }
    public HashMap<DatapathId, Integer> getCosts() {
        return costs;
    }
}
