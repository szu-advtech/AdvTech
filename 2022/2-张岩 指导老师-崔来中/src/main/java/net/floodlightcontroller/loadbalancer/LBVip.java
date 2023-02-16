package net.floodlightcontroller.loadbalancer;
import java.util.ArrayList;
import org.projectfloodlight.openflow.types.MacAddress;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import net.floodlightcontroller.loadbalancer.LoadBalancer.IPClient;
@JsonSerialize(using=LBVipSerializer.class)
public class LBVip {
    protected String id;    
    protected String name;
    protected String tenantId;
    protected String netId;
    protected int address;
    protected byte protocol;
    protected short lbMethod;
    protected short port;
    protected ArrayList<String> pools;
    protected boolean sessionPersistence;
    protected int connectionLimit;
    protected short adminState;
    protected short status;
    protected MacAddress proxyMac;
    public static String LB_PROXY_MAC= "12:34:56:78:90:12";
    public LBVip() {
        this.name = null;
        this.tenantId = null;
        this.netId = null;
        this.address = 0;
        this.protocol = 0;
        this.lbMethod = 0;
        this.port = 0;
        this.pools = new ArrayList<String>();
        this.sessionPersistence = false;
        this.connectionLimit = 0;
        this.address = 0;
        this.status = 0;
        this.proxyMac = MacAddress.of(LB_PROXY_MAC);
    }
    public String pickPool(IPClient client) {
        if (pools.size() > 0)
            return pools.get(0);
        else
            return null;
    }
}
