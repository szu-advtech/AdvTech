package net.floodlightcontroller.core;
import java.util.concurrent.ConcurrentHashMap;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IOFMessageFilterManagerService extends IFloodlightService {
	String setupFilter(String sid, ConcurrentHashMap<String, String> f,
			int deltaInMilliSeconds);
}
