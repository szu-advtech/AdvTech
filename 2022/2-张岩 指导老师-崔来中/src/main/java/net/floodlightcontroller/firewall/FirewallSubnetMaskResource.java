package net.floodlightcontroller.firewall;
import java.io.IOException;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.MappingJsonFactory;
import org.restlet.resource.Post;
import org.restlet.resource.Get;
import org.restlet.data.Status;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class FirewallSubnetMaskResource extends FirewallResourceBase {
	private static final Logger log = LoggerFactory.getLogger(FirewallSubnetMaskResource.class);
	@Get("json")
	public Object handleRequest() {
		IFirewallService firewall = getFirewallService();
		return "{\"subnet-mask\":\"" + firewall.getSubnetMask() + "\"}";
	}
	@Post
	public String handlePost(String fmJson) {
		IFirewallService firewall = getFirewallService();
		String newMask;
		try {
			newMask = jsonExtractSubnetMask(fmJson);
		} catch (IOException e) {
			log.error("Error parsing new subnet mask: " + fmJson, e);
			setStatus(Status.CLIENT_ERROR_BAD_REQUEST);
			return "{\"status\" : \"Error! Could not parse new subnet mask, see log for details.\"}";
		}
		firewall.setSubnetMask(newMask);
		setStatus(Status.SUCCESS_OK);
		return ("{\"status\" : \"subnet mask set\"}");
	}
	public static String jsonExtractSubnetMask(String fmJson) throws IOException {
		String subnet_mask = "";
		MappingJsonFactory f = new MappingJsonFactory();
		JsonParser jp;
		try {
			jp = f.createParser(fmJson);
		} catch (JsonParseException e) {
			throw new IOException(e);
		}
		jp.nextToken();
		if (jp.getCurrentToken() != JsonToken.START_OBJECT) {
			throw new IOException("Expected START_OBJECT");
		}
		while (jp.nextToken() != JsonToken.END_OBJECT) {
			if (jp.getCurrentToken() != JsonToken.FIELD_NAME) {
				throw new IOException("Expected FIELD_NAME");
			}
			String n = jp.getCurrentName();
			jp.nextToken();
			if (jp.getText().equals(""))
				continue;
			if (n == "subnet-mask") {
				subnet_mask = jp.getText();
				break;
			}
		}
		return subnet_mask;
	}
}
