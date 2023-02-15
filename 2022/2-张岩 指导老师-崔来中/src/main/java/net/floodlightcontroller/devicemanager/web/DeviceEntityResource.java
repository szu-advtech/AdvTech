package net.floodlightcontroller.devicemanager.web;
import java.util.Iterator;
import net.floodlightcontroller.devicemanager.IDevice;
import net.floodlightcontroller.devicemanager.internal.Device;
import net.floodlightcontroller.devicemanager.internal.Entity;
import org.restlet.resource.Get;
public class DeviceEntityResource extends AbstractDeviceResource {
    @Get("json")
    public Iterator<Entity[]> getDeviceEntities() {
        final Iterator<? extends IDevice> devices = super.getDevices();
        return new Iterator<Entity[]>() {
            @Override
            public boolean hasNext() {
                return devices.hasNext();
            }
            @Override
            public Entity[] next() {
                Device d = (Device)devices.next();
                return d.getEntities();
            }
            @Override
            public void remove() {
                throw new UnsupportedOperationException();
            }
        };
    }
}
