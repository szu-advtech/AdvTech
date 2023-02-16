package net.floodlightcontroller.firewall;
import java.util.List;
import java.util.Map;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IFirewallService extends IFloodlightService {
    public void enableFirewall(boolean enable);
    public boolean isEnabled();
    public List<FirewallRule> getRules();
    public String getSubnetMask();
    public void setSubnetMask(String newMask);
    public List<Map<String, Object>> getStorageRules();
    public void addRule(FirewallRule rule);
    public void deleteRule(int ruleid);
}
