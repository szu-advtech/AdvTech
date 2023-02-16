package net.floodlightcontroller.virtualnetwork;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import org.projectfloodlight.openflow.types.MacAddress;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
@JsonSerialize(using=VirtualNetworkSerializer.class)
public class VirtualNetwork{
    public VirtualNetwork(String name, String guid) {
        this.name = name;
        this.guid = guid;
        this.gateway = null;
		this.portToMac = new ConcurrentHashMap<String,MacAddress>();
        return;        
    }
    public void setName(String name){
        this.name = name;
        return;                
    }
    public void setGateway(String gateway){
        this.gateway = gateway;
        return;                
    }
    public void addHost(String port, MacAddress host){
        return;         
    }
    public boolean removeHost(MacAddress host){
		for (Entry<String, MacAddress> entry : this.portToMac.entrySet()) {
			if (entry.getValue().equals(host)){
				this.portToMac.remove(entry.getKey());
				return true;
			}
		}
		return false;
    }
    public void clearHosts(){
		this.portToMac.clear();
    }
}
