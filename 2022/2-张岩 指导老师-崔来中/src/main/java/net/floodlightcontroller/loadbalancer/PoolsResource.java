package net.floodlightcontroller.loadbalancer;
import java.io.IOException;
import java.util.Collection;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.MappingJsonFactory;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.restlet.resource.Delete;
import org.restlet.resource.Get;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class PoolsResource extends ServerResource {
    protected static Logger log = LoggerFactory.getLogger(PoolsResource.class);
    @Get("json")
    public Collection <LBPool> retrieve() {
        ILoadBalancerService lbs =
                (ILoadBalancerService)getContext().getAttributes().
                    get(ILoadBalancerService.class.getCanonicalName());
        String poolId = (String) getRequestAttributes().get("pool");
        if (poolId!=null)
            return lbs.listPool(poolId);
        else        
            return lbs.listPools();               
    }
    @Put
    @Post
    public LBPool createPool(String postData) {        
        LBPool pool=null;
        try {
            pool=jsonToPool(postData);
        } catch (IOException e) {
            log.error("Could not parse JSON {}", e.getMessage());
        }
        ILoadBalancerService lbs =
                (ILoadBalancerService)getContext().getAttributes().
                    get(ILoadBalancerService.class.getCanonicalName());
        String poolId = (String) getRequestAttributes().get("pool");
        if (poolId != null)
            return lbs.updatePool(pool);
        else        
            return lbs.createPool(pool);
    }
    @Delete
    public int removePool() {
        String poolId = (String) getRequestAttributes().get("pool");
        ILoadBalancerService lbs =
                (ILoadBalancerService)getContext().getAttributes().
                    get(ILoadBalancerService.class.getCanonicalName());
        return lbs.removePool(poolId);
    }
    protected LBPool jsonToPool(String json) throws IOException {
        if (json==null) return null;
        MappingJsonFactory f = new MappingJsonFactory();
        JsonParser jp;
        LBPool pool = new LBPool();
        try {
            jp = f.createParser(json);
        } catch (JsonParseException e) {
            throw new IOException(e);
        }
        jp.nextToken();
        if (jp.getCurrentToken() != JsonToken.START_OBJECT) {
            throw new IOException("Expected START_OBJECT");
        }
        while (jp.nextToken() != JsonToken.END_OBJECT) {
            if (jp.getCurrentToken() != JsonToken.FIELD_NAME) {
                throw new IOException("Expected FIELD_NAME");
            }
            String n = jp.getCurrentName();
            jp.nextToken();
            if (jp.getText().equals("")) 
                continue;
            if (n.equals("id")) {
                pool.id = jp.getText();
                continue;
            } 
            if (n.equals("tenant_id")) {
                pool.tenantId = jp.getText();
                continue;
            } 
            if (n.equals("name")) {
                pool.name = jp.getText();
                continue;
            }
            if (n.equals("network_id")) {
                pool.netId = jp.getText();
                continue;
            }
            if (n.equals("lb_method")) {
                pool.lbMethod = Short.parseShort(jp.getText());
                continue;
            }
            if (n.equals("protocol")) {
                String tmp = jp.getText();
                if (tmp.equalsIgnoreCase("TCP")) {
                    pool.protocol = (byte) IpProtocol.TCP.getIpProtocolNumber();
                } else if (tmp.equalsIgnoreCase("UDP")) {
                    pool.protocol = (byte) IpProtocol.UDP.getIpProtocolNumber();
                } else if (tmp.equalsIgnoreCase("ICMP")) {
                    pool.protocol = (byte) IpProtocol.ICMP.getIpProtocolNumber();
                } 
                continue;
            }                    
            if (n.equals("vip_id")) {
                pool.vipId = jp.getText();
                continue;
            } 
            log.warn("Unrecognized field {} in " +
                    "parsing Pools", 
                    jp.getText());
        }
        jp.close();
        return pool;
    }
}
