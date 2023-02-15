package net.floodlightcontroller.virtualnetwork;
import org.restlet.data.Status;
import org.restlet.resource.Delete;
import org.restlet.resource.Get;
import org.restlet.resource.Post;
import org.restlet.resource.Put;
import org.restlet.resource.ServerResource;
public class NoOp extends ServerResource {
    @Get
    @Put
    @Post
    @Delete
    public String noOp(String postdata) {
        setStatus(Status.SUCCESS_OK);
        return "{\"status\":\"ok\"}";
    }
}
