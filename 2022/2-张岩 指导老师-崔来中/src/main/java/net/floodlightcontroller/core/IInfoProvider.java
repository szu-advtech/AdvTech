package net.floodlightcontroller.core;
import java.util.Map;
public interface IInfoProvider {
    public Map<String, Object> getInfo(String type);
}
