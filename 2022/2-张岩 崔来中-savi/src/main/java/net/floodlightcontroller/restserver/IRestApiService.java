package net.floodlightcontroller.restserver;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IRestApiService extends IFloodlightService {
    public void addRestletRoutable(RestletRoutable routable);
    public void run();
}
