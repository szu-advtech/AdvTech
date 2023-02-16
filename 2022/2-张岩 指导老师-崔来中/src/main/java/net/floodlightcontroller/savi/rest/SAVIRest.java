package net.floodlightcontroller.savi.rest;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.restlet.routing.Router;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.MappingJsonFactory;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.restserver.RestletRoutable;
import net.floodlightcontroller.savi.binding.Binding;
import net.floodlightcontroller.savi.service.SAVIProviderService;
public class SAVIRest extends ServerResource {
	protected static Logger log = LoggerFactory.getLogger(SAVIRest.class);
	public static final String ADD_SECURITY_PORT_TYPE = "add_security_port";
	public static final String DEL_SECURITY_PORT_TYPE = "del_security_port";
	public static final String GET_SECURITY_PORT_TYPE = "get_security_port";
	public static final String GET_BINDING_TYPE = "get_binding";
	public static final String GET_GROUP_BINDING = "get_group_binding";
	public static final String TYPE = "type";
	public static final String SWITCH_DPID = "dpid";
	public static final String PORT_NUM = "port";
	public static final String MAC = "mac";
	public static final String IP = "ip";
	public static final String IPv6 = "ipv6";
	public class SAVIRoutable implements RestletRoutable{
		@Override
		public Restlet getRestlet(Context context) {
			Router router = new Router(context);
			router.attach("/config", SAVIRest.class);
			return router;
		}
		@Override
		public String basePath() {
			return "/savi";
		}	
	}
	public String post(String json){
		Map<String, String> jsonMap = jsonToStringMap(json);
		if(jsonMap == null){
			return "{FAIL}";
		}
		switch(jsonMap.get(TYPE)){
		case ADD_SECURITY_PORT_TYPE:
			break;
		case DEL_SECURITY_PORT_TYPE:
			break;
		case GET_SECURITY_PORT_TYPE:
			break;
		case GET_BINDING_TYPE:
			return doGetBinding();
		}
		return null;
	@Post
	public Object post(String json){
		log.info(json);
		Map<String, String> jsonMap = jsonToStringMap(json);
		if(jsonMap == null){
			return null;
		}
		switch(jsonMap.get(TYPE)){
		case ADD_SECURITY_PORT_TYPE:
			break;
		case DEL_SECURITY_PORT_TYPE:
			break;
		case GET_SECURITY_PORT_TYPE:
			return doGetSecurityPort();
		case GET_BINDING_TYPE:
			return doGetBinding();
		case GET_GROUP_BINDING:
			return doGetBindingGroup();
		}
		return null;
	}
	class BindItem{
		private String swport;
		private String mac;
		private String time;
		private String ip;
		public BindItem(String swport,String mac,String time,String ip){
			this.swport=swport;
			this.mac=mac;
			this.time=time;
			this.ip=ip;
		}
		public String getSwport() {
			return swport;
		}
		public void setSwport(String swport) {
			this.swport = swport;
		}
		public String getMac() {
			return mac;
		}
		public void setMac(String mac) {
			this.mac = mac;
		}
		public String getTime() {
			return time;
		}
		public void setTime(String time) {
			this.time = time;
		}
		public String getIp() {
			return ip;
		}
		public void setIp(String ip) {
			this.ip = ip;
		}
	}
	protected synchronized String doAddSecurityPort(Map<String, String> jsonMap){
		SAVIProviderService providerService = (SAVIProviderService)getContext().getAttributes().get(SAVIProviderService.class.getCanonicalName());
		if(jsonMap.containsKey(SWITCH_DPID)&&jsonMap.containsKey(PORT_NUM)){
			DatapathId dpid = DatapathId.of(jsonMap.get(SWITCH_DPID));
			OFPort port = OFPort.ofInt(Integer.valueOf(jsonMap.get(PORT_NUM)));
			providerService.addSecurityPort(new SwitchPort(dpid,port));
		}
		return "{SUCCESS}";
	}
	protected synchronized String doDelSecurityPort(Map<String, String> jsonMap){	
		SAVIProviderService providerService = (SAVIProviderService)getContext().getAttributes().get(SAVIProviderService.class.getCanonicalName());
		if(jsonMap.containsKey(SWITCH_DPID)&&jsonMap.containsKey(PORT_NUM)){
			DatapathId dpid = DatapathId.of(jsonMap.get(SWITCH_DPID));
			OFPort port = OFPort.ofInt(Integer.valueOf(jsonMap.get(PORT_NUM)));
			providerService.delSecurityPort(new SwitchPort(dpid,port));
		}
		return "{SUCCESS}";
	}
		SAVIProviderService providerService = (SAVIProviderService)getContext().getAttributes().get(SAVIProviderService.class.getCanonicalName());
		List<Binding<?>> bindings = providerService.getBindings();
		int i = 0;
		String ret = "[";
		for(;i<bindings.size()-1;i++){
			ret += bindings.get(i).toString()+",";
		}
		if(i>0){
			ret += bindings.get(i).toString();
		}
		return ret;
	protected synchronized List<BindItem> doGetBinding(){
		SAVIProviderService providerService = 
				(SAVIProviderService)getContext()
				.getAttributes().get(SAVIProviderService.class.getCanonicalName());
		List<Binding<?>> bindings = providerService.getBindings();
		int i = 0;
		List<BindItem> result=new ArrayList<>();
		for(;i<bindings.size();i++){
			BindItem temp=new BindItem(bindings.get(i).getSwitchPort().getSwitchDPID().toString()
					+"-"+bindings.get(i).getSwitchPort().getPort().toString(), 
					bindings.get(i).getMacAddress().toString(), 
					bindings.get(i).getBindingTime()+"", 
					bindings.get(i).getAddress().toString());
			result.add(temp);
		}
		return result;
	}
	protected synchronized Map<DatapathId, List<BindItem>> doGetBindingGroup(){
		SAVIProviderService providerService = 
				(SAVIProviderService)getContext()
				.getAttributes().get(SAVIProviderService.class.getCanonicalName());
		List<Binding<?>> bindings = providerService.getBindings();
		Map<DatapathId, List<BindItem>> result = new HashMap<>();
		int i = 0;
		for(;i<bindings.size();i++){
			DatapathId temp = bindings.get(i).getSwitchPort().getSwitchDPID();
			BindItem tempBind=new BindItem(bindings.get(i).getSwitchPort().getPort().toString(), 
					bindings.get(i).getMacAddress().toString(), 
					bindings.get(i).getBindingTime()+"", 
					bindings.get(i).getAddress().toString());
			List<BindItem> bindlists = result.get(temp);
			if(bindlists == null){
				bindlists = new ArrayList<>();
				bindlists.add(tempBind);
				result.put(temp, bindlists);
			}
			else {
				bindlists.add(tempBind);
			}
		}
		return result;
	}
		SAVIProviderService providerService = (SAVIProviderService)getContext().getAttributes().get(SAVIProviderService.class.getCanonicalName());
		int i = 0;
		String ret = "[";
		SwitchPort[] set = (SwitchPort[]) providerService.getSecurityPorts().toArray();
		for(;i<set.length-1;i++){
			ret += set[i].toString() + ","; 
		}
		if(i>0){
			ret += set[i].toString();
		}
		ret += "]";
		return ret;
	protected synchronized Object doGetSecurityPort(){
		SAVIProviderService providerService = (SAVIProviderService)getContext().getAttributes().get(SAVIProviderService.class.getCanonicalName());
		return providerService.getSecurityPorts().toArray();
	}
	public Map<String, String> jsonToStringMap(String json){
		ObjectMapper mapper = new ObjectMapper();    
        try {   
            Map<String,String> m = mapper.readValue(json, Map.class);  
            return m;
        } catch (JsonParseException e) {  
            e.printStackTrace();  
        } catch (JsonMappingException e) {  
            e.printStackTrace();  
        } catch (IOException e) {  
            e.printStackTrace();  
        return null;
	public Map<String, String> jsonToStringMap(String json){
		Map<String, String> jsonMap = new HashMap<>();
		JsonParser jp;
		MappingJsonFactory f = new MappingJsonFactory();
		try {
			jp = f.createJsonParser(json);
		}
		catch(Exception e){
			e.printStackTrace();
			return null;
		}
		try{
			jp.nextToken();
			if(jp.getCurrentToken() != JsonToken.START_OBJECT){
				return null;
			}
			while(jp.nextToken()!=JsonToken.END_OBJECT){
				String name = jp.getCurrentName();
				jp.nextToken();
				jsonMap.put(name, jp.getCurrentName());
			}
			return jsonMap;
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return null;
	public Map<String, String> jsonToStringMap(String json){
		Map<String, String> jsonMap=new HashMap<>();
		MappingJsonFactory f = new MappingJsonFactory();
		JsonParser jp = null;
		try {
			try {
				jp = f.createParser(json);
			} catch (IOException e) {
				e.printStackTrace();
			}
			jp.nextToken();
			if (jp.getCurrentToken() != JsonToken.START_OBJECT) {
				throw new IOException("Expected START_OBJECT");
			}
			while (jp.nextToken() != JsonToken.END_OBJECT) {
				if (jp.getCurrentToken() != JsonToken.FIELD_NAME) {
					throw new IOException("Expected FIELD_NAME");
				}
				String key = jp.getCurrentName().toLowerCase();
				jp.nextToken();
				jsonMap.put(key, jp.getText());
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		return jsonMap;
	}
}
