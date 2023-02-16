package net.floodlightcontroller.linkdiscovery;
import java.util.List;
public interface ILinkDiscoveryListener extends ILinkDiscovery{
    public void linkDiscoveryUpdate(LDUpdate update);
    public void linkDiscoveryUpdate(List<LDUpdate> updateList);
}
