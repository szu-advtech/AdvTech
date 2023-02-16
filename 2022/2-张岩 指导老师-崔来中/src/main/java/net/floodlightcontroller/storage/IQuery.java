package net.floodlightcontroller.storage;
public interface IQuery {
    String getTableName();
    void setParameter(String name, Object value);
}
