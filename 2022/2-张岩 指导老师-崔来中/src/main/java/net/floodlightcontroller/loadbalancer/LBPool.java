package net.floodlightcontroller.loadbalancer;
import java.util.ArrayList;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import net.floodlightcontroller.loadbalancer.LoadBalancer.IPClient;
@JsonSerialize(using=LBPoolSerializer.class)
public class LBPool {
    protected String id;
    protected String name;
    protected String tenantId;
    protected String netId;
    protected short lbMethod;
    protected byte protocol;
    protected ArrayList<String> members;
    protected ArrayList<String> monitors;
    protected short adminState;
    protected short status;
    protected String vipId;
    protected int previousMemberIndex;
    public LBPool() {
        name = null;
        tenantId = null;
        netId = null;
        lbMethod = 0;
        protocol = 0;
        members = new ArrayList<String>();
        monitors = new ArrayList<String>();
        adminState = 0;
        status = 0;
        previousMemberIndex = -1;
    }
    public String pickMember(IPClient client) {
        if (members.size() > 0) {
            previousMemberIndex = (previousMemberIndex + 1) % members.size();
            return members.get(previousMemberIndex);
        } else {
            return null;
        }
    }
}
