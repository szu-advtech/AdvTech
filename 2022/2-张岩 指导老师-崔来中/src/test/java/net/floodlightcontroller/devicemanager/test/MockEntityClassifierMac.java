package net.floodlightcontroller.devicemanager.test;
import static net.floodlightcontroller.devicemanager.IDeviceService.DeviceField.MAC;
import static net.floodlightcontroller.devicemanager.IDeviceService.DeviceField.PORT;
import static net.floodlightcontroller.devicemanager.IDeviceService.DeviceField.SWITCH;
import static net.floodlightcontroller.devicemanager.IDeviceService.DeviceField.VLAN;
import java.util.EnumSet;
import org.projectfloodlight.openflow.types.DatapathId;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.IEntityClass;
import net.floodlightcontroller.devicemanager.IDeviceService.DeviceField;
import net.floodlightcontroller.devicemanager.internal.DefaultEntityClassifier;
import net.floodlightcontroller.devicemanager.internal.Entity;
public class MockEntityClassifierMac extends DefaultEntityClassifier {
    public static class TestEntityClassMac implements IEntityClass {
        protected String name;
        public TestEntityClassMac(String name) {
            this.name = name;
        }
        @Override
        public EnumSet<DeviceField> getKeyFields() {
            return EnumSet.of(MAC, VLAN);
        }
        @Override
        public String getName() {
            return name;
        }
    }
    public static IEntityClass testECMac1 = 
            new MockEntityClassifierMac.TestEntityClassMac("testECMac1");
    public static IEntityClass testECMac2 = 
            new MockEntityClassifierMac.TestEntityClassMac("testECMac2");
    @Override
    public IEntityClass classifyEntity(Entity entity) {
        if (entity.getSwitchDPID() == null) {
            throw new IllegalArgumentException("Not all key fields specified."
                    + " Required fields: " + getKeyFields());
        } else if (entity.getSwitchDPID().equals(DatapathId.of(1))) {
            return testECMac1;
        } else if (entity.getSwitchDPID().equals(DatapathId.of(2))) {
            return testECMac2;
        } else if (entity.getSwitchDPID().equals(DatapathId.of(-1))) {
            return null;
        }
        return DefaultEntityClassifier.entityClass;
    }
    @Override
    public EnumSet<IDeviceService.DeviceField> getKeyFields() {
        return EnumSet.of(MAC, VLAN, SWITCH, PORT);
    }
}