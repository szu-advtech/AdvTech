package net.floodlightcontroller.core;
import java.util.Map;
public interface IHAListener extends IListener<HAListenerTypeMarker> {
    public void transitionToActive();
    public void transitionToStandby();
    public void controllerNodeIPsChanged(Map<String, String> curControllerNodeIPs,
    									Map<String, String> addedControllerNodeIPs,
    									Map<String, String> removedControllerNodeIPs);
}
