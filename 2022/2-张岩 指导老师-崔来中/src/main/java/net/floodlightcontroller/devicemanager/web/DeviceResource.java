package net.floodlightcontroller.devicemanager.web;
import java.util.Iterator;
import net.floodlightcontroller.devicemanager.IDevice;
import org.restlet.resource.Get;
public class DeviceResource extends AbstractDeviceResource {
    @Get("json")
    public Iterator<? extends IDevice> getDevices() {
        return super.getDevices();
    }
}
