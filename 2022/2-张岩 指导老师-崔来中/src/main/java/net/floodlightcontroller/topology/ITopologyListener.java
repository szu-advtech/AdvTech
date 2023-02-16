package net.floodlightcontroller.topology;
import java.util.List;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LDUpdate;
public interface ITopologyListener {
    void topologyChanged(List<LDUpdate> linkUpdates);
}
