package net.floodlightcontroller.storage.memory;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
public class MemoryTable {
    private String tableName;
    private Map<Object,Map<String,Object>> rowMap;
    private int nextId;
    MemoryTable(String tableName) {
        this.tableName = tableName;
        rowMap = new TreeMap<Object,Map<String,Object>>();
        nextId = 0;
    }
    String getTableName() {
        return tableName;
    }
    Collection<Map<String,Object>> getAllRows() {
        return rowMap.values();
    }
    Map<String,Object> getRow(Object key) {
        Map<String,Object> row = rowMap.get(key);
        return row;
    }
    Map<String,Object> newRow(Object key) {
        Map<String,Object> row = new HashMap<String, Object>();
        row.put("id", key);
        rowMap.put(key, row);
        return row;
    }
    void insertRow(Object key, Map<String,Object> rowValues) {
        assert(key != null);
        rowMap.put(key, rowValues);
    }
    void deleteRow(Object rowKey) {
        rowMap.remove(rowKey);
    }
    Integer getNextId() {
        return new Integer(++nextId);
    }
}
