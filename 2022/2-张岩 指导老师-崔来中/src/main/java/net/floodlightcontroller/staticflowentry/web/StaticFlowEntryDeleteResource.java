package net.floodlightcontroller.staticflowentry.web;
import java.io.IOException;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.staticflowentry.StaticFlowEntries;
import net.floodlightcontroller.staticflowentry.StaticFlowEntryPusher;
import net.floodlightcontroller.storage.IStorageSourceService;
public class StaticFlowEntryDeleteResource extends ServerResource {
    protected static Logger log = LoggerFactory.getLogger(StaticFlowEntryDeleteResource.class);
    @Post
    public String del(String fmJson) {
        IStorageSourceService storageSource =
                (IStorageSourceService)getContext().getAttributes().
                    get(IStorageSourceService.class.getCanonicalName());
        String fmName = null;
        if (fmJson == null) {
            return "{\"status\" : \"Error! No data posted.\"}";
        }
        try {
            fmName = StaticFlowEntries.getEntryNameFromJson(fmJson);
            if (fmName == null) {
                return "{\"status\" : \"Error deleting entry, no name provided\"}";
            }
        } catch (IOException e) {
            log.error("Error deleting flow mod request: " + fmJson, e);
            return "{\"status\" : \"Error deleting entry, see log for details\"}";
        }
        storageSource.deleteRowAsync(StaticFlowEntryPusher.TABLE_NAME, fmName);
        return "{\"status\" : \"Entry " + fmName + " deleted\"}";
    }
}
