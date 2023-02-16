package net.floodlightcontroller.restserver;
import org.restlet.Context;
import org.restlet.Restlet;
public interface RestletRoutable {
    Restlet getRestlet(Context context);
    String basePath();
}
