package net.floodlightcontroller.servicechaining;
import net.floodlightcontroller.core.FloodlightContextStore;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IServiceChainingService extends IFloodlightService {
    public static final FloodlightContextStore<String> scStore =
        new FloodlightContextStore<String>();
    public ServiceChain getServiceChainBySrcBVS(String bvsName);
    public ServiceChain getServiceChainByDstBVS(String bvsName);
}
