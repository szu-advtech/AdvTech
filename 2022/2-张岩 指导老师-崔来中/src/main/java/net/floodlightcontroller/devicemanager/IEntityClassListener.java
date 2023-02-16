package net.floodlightcontroller.devicemanager;
import java.util.Set;
public interface IEntityClassListener {
    public void entityClassChanged(Set<String> entityClassNames);
}
