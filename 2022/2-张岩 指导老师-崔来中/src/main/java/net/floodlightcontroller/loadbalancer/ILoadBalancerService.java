package net.floodlightcontroller.loadbalancer;
import java.util.Collection;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface ILoadBalancerService extends IFloodlightService {
    public Collection<LBVip> listVips();
    public Collection<LBVip> listVip(String vipId);
    public LBVip createVip(LBVip vip);
    public LBVip updateVip(LBVip vip);
    public int removeVip(String vipId);
    public Collection<LBPool> listPools();
    public Collection<LBPool> listPool(String poolId);
    public LBPool createPool(LBPool pool);
    public LBPool updatePool(LBPool pool);
    public int removePool(String poolId);
    public Collection<LBMember> listMembers();
    public Collection<LBMember> listMember(String memberId);
    public Collection<LBMember> listMembersByPool(String poolId);
    public LBMember createMember(LBMember member);
    public LBMember updateMember(LBMember member);
    public int removeMember(String memberId);
    public Collection<LBMonitor> listMonitors();
    public Collection<LBMonitor> listMonitor(String monitorId);
    public LBMonitor createMonitor(LBMonitor monitor);
    public LBMonitor updateMonitor(LBMonitor monitor);
    public int removeMonitor(String monitorId);
}
