package net.floodlightcontroller.storage;
import java.util.Set;
public interface IStorageSourceListener {
    public void rowsModified(String tableName, Set<Object> rowKeys);
    public void rowsDeleted(String tableName, Set<Object> rowKeys);
}
