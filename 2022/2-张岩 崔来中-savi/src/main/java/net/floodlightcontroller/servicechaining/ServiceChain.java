package net.floodlightcontroller.servicechaining;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
public class ServiceChain {
    private String tenant;
    private String name;
    private String srcBvsName;
    private String dstBvsName;
    private String description;
    private List<ServiceNode> nodes;
    public ServiceChain(String tenant, String name, String description,
            String srcBvsName, String dstBvsName) {
        this.tenant = tenant;
        this.name = name;
        this.description = description;
        this.srcBvsName = srcBvsName;
        this.dstBvsName = dstBvsName;
        this.nodes = new ArrayList<ServiceNode>();
    }
    public String getTenant() {
        return tenant;
    }
    public String getName() {
        return name;
    }
    public String getDescription() {
        return description;
    }
    public String getSourceBvs() {
        return srcBvsName;
    }
    public String getDestinationBvs() {
        return dstBvsName;
    }
    public List<ServiceNode> getServiceNodes() {
        return Collections.unmodifiableList(nodes);
    }
    public boolean addNode(ServiceNode node) {
        try {
            return nodes.add(node);
        } catch (Exception e) {
            return false;
        }
    }
    public boolean removeNode(ServiceNode node) {
        try {
            return nodes.remove(node);
        } catch (Exception e) {
            return false;
        }
    }
    @Override
    public String toString() {
        return "ServiceChain [tenant=" + tenant + ", name=" + name
                + ", srcBvsName=" + srcBvsName + ", dstBvsName=" + dstBvsName
                + ", description=" + description + "]";
    }
}
