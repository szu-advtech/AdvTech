package net.floodlightcontroller.savi.action;
import java.util.HashMap;
import java.util.Map;
public class ConfigureAction extends Action {
	Map<String, Object> configureMap;
	public ConfigureAction() {
		this.type = ActionType.CONFIGURE;
		configureMap = new HashMap<>();
	}
	public ConfigureAction(Map<String, Object> map){
		this.type = ActionType.CONFIGURE;
		configureMap = map;
	}
	public void addConfigureEntry(String string, Object entry){
		this.configureMap.put(string, entry);
	}
}
