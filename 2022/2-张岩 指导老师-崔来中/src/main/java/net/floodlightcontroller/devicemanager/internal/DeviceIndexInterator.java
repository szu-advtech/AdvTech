package net.floodlightcontroller.devicemanager.internal;
import java.util.Iterator;
public class DeviceIndexInterator implements Iterator<Device> {
    private DeviceManagerImpl deviceManager;
    private Iterator<Long> subIterator;
    public DeviceIndexInterator(DeviceManagerImpl deviceManager,
                                Iterator<Long> subIterator) {
        super();
        this.deviceManager = deviceManager;
        this.subIterator = subIterator;
    }
    @Override
    public boolean hasNext() {
        return subIterator.hasNext();
    }
    @Override
    public Device next() {
        Long next = subIterator.next();
        return deviceManager.deviceMap.get(next);
    }
    @Override
    public void remove() {
        subIterator.remove();
    }
}