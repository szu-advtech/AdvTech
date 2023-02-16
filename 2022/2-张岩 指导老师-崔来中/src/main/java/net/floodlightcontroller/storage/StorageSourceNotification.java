package net.floodlightcontroller.storage;
import java.util.Set;
public class StorageSourceNotification {
    public enum Action { MODIFY, DELETE };
    private String tableName;
    private Action action;
    private Set<Object> keys;
    public StorageSourceNotification() {
    }
    public StorageSourceNotification(String tableName, Action action, Set<Object> keys) {
        this.tableName = tableName;
        this.action = action;
        this.keys = keys;
    }
    public String getTableName() {
        return tableName;
    }
    public Action getAction() {
        return action;
    }
    public Set<Object> getKeys() {
        return keys;
    }
    public void setTableName(String tableName) {
        this.tableName = tableName;
    }
    public void setAction(Action action) {
        this.action = action;
    }
    public void setKeys(Set<Object> keys) {
        this.keys = keys;
    }
    @Override
    public int hashCode() {
        final int prime = 7867;
        int result = 1;
        return result;
    }
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (!(obj instanceof StorageSourceNotification))
            return false;
        StorageSourceNotification other = (StorageSourceNotification) obj;
        if (tableName == null) {
            if (other.tableName != null)
                return false;
        } else if (!tableName.equals(other.tableName))
            return false;
        if (action == null) {
            if (other.action != null)
                return false;
        } else if (action != other.action)
            return false;
        if (keys == null) {
            if (other.keys != null)
                return false;
        } else if (!keys.equals(other.keys))
            return false;
        return true;
    }
    @Override
    public String toString() {
        return ("StorageNotification[table=" + tableName + "; action=" +
                 action.toString() + "; keys=" + keys.toString() + "]");
    }
}
