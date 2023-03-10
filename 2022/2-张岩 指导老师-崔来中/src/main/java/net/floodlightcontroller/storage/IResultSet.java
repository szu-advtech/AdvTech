package net.floodlightcontroller.storage;
import java.util.Date;
import java.util.Map;
public interface IResultSet extends Iterable<IResultSet> {
    public void close();
    public boolean next();
    public void save();
    public Map<String,Object> getRow();
    public void deleteRow();
    public boolean containsColumn(String columnName);
    public String getString(String columnName);
    public short getShort(String columnName);
    public int getInt(String columnName);
    public long getLong(String columnName);
    public float getFloat(String columnName);
    public double getDouble(String columnName);
    public boolean getBoolean(String columnName);
    public byte getByte(String columnName);
    public byte[] getByteArray(String columnName);
    public Date getDate(String columnName);
    public Short getShortObject(String columnName);
    public Integer getIntegerObject(String columnName);
    public Long getLongObject(String columnName);
    public Float getFloatObject(String columnName);
    public Double getDoubleObject(String columnName);
    public Boolean getBooleanObject(String columnName);
    public Byte getByteObject(String columnName);
    public boolean isNull(String columnName);
    public void setString(String columnName, String value);
    public void setShort(String columnName, short value);
    public void setInt(String columnName, int value);
    public void setLong(String columnName, long value);
    public void setFloat(String columnName, float value);
    public void setDouble(String columnName, double value);
    public void setBoolean(String columnName, boolean value);
    public void setByte(String columnName, byte value);
    public void setByteArray(String columnName, byte[] byteArray);
    public void setDate(String columnName, Date date);
    public void setShortObject(String columnName, Short value);
    public void setIntegerObject(String columnName, Integer value);
    public void setLongObject(String columnName, Long value);
    public void setFloatObject(String columnName, Float value);
    public void setDoubleObject(String columnName, Double value);
    public void setBooleanObject(String columnName, Boolean value);
    public void setByteObject(String columnName, Byte value);
    public void setNull(String columnName);
}
