package net.floodlightcontroller.virtualnetwork;
import java.util.Collection;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IVirtualNetworkService extends IFloodlightService {
    public void createNetwork(String guid, String network, IPv4Address gateway);
    public void deleteNetwork(String guid);
    public void addHost(MacAddress mac, String network, String port); 
    public void deleteHost(MacAddress mac, String port);
    public Collection <VirtualNetwork> listNetworks();
}
