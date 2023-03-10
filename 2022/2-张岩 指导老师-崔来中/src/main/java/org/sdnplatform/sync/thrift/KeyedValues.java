package org.sdnplatform.sync.thrift;
import org.apache.thrift.scheme.IScheme;
import org.apache.thrift.scheme.SchemeFactory;
import org.apache.thrift.scheme.StandardScheme;
import org.apache.thrift.scheme.TupleScheme;
import org.apache.thrift.protocol.TTupleProtocol;
import org.apache.thrift.protocol.TProtocolException;
import org.apache.thrift.EncodingUtils;
import org.apache.thrift.TException;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.EnumMap;
import java.util.Set;
import java.util.HashSet;
import java.util.EnumSet;
import java.util.Collections;
import java.util.BitSet;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
@SuppressWarnings("all") public class KeyedValues implements org.apache.thrift.TBase<KeyedValues, KeyedValues._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("KeyedValues");
  private static final org.apache.thrift.protocol.TField KEY_FIELD_DESC = new org.apache.thrift.protocol.TField("key", org.apache.thrift.protocol.TType.STRING, (short)1);
  private static final org.apache.thrift.protocol.TField VALUES_FIELD_DESC = new org.apache.thrift.protocol.TField("values", org.apache.thrift.protocol.TType.LIST, (short)2);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new KeyedValuesStandardSchemeFactory());
    schemes.put(TupleScheme.class, new KeyedValuesTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    KEY((short)1, "key"),
    VALUES((short)2, "values");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return KEY;
          return VALUES;
        default:
          return null;
      }
    }
    public static _Fields findByThriftIdOrThrow(int fieldId) {
      _Fields fields = findByThriftId(fieldId);
      if (fields == null) throw new IllegalArgumentException("Field " + fieldId + " doesn't exist!");
      return fields;
    }
    public static _Fields findByName(String name) {
      return byName.get(name);
    }
    private final short _thriftId;
    private final String _fieldName;
    _Fields(short thriftId, String fieldName) {
      _thriftId = thriftId;
      _fieldName = fieldName;
    }
    public short getThriftFieldId() {
      return _thriftId;
    }
    public String getFieldName() {
      return _fieldName;
    }
  }
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.KEY, new org.apache.thrift.meta_data.FieldMetaData("key", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING        , true)));
    tmpMap.put(_Fields.VALUES, new org.apache.thrift.meta_data.FieldMetaData("values", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, VersionedValue.class))));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(KeyedValues.class, metaDataMap);
  }
  public KeyedValues() {
  }
  public KeyedValues(
    ByteBuffer key,
    List<VersionedValue> values)
  {
    this();
    this.key = key;
    this.values = values;
  }
  public KeyedValues(KeyedValues other) {
    if (other.isSetKey()) {
      this.key = org.apache.thrift.TBaseHelper.copyBinary(other.key);
;
    }
    if (other.isSetValues()) {
      List<VersionedValue> __this__values = new ArrayList<VersionedValue>();
      for (VersionedValue other_element : other.values) {
        __this__values.add(new VersionedValue(other_element));
      }
      this.values = __this__values;
    }
  }
  public KeyedValues deepCopy() {
    return new KeyedValues(this);
  }
  @Override
  public void clear() {
    this.key = null;
    this.values = null;
  }
  public byte[] getKey() {
    setKey(org.apache.thrift.TBaseHelper.rightSize(key));
    return key == null ? null : key.array();
  }
  public ByteBuffer bufferForKey() {
    return key;
  }
  public KeyedValues setKey(byte[] key) {
    setKey(key == null ? (ByteBuffer)null : ByteBuffer.wrap(key));
    return this;
  }
  public KeyedValues setKey(ByteBuffer key) {
    this.key = key;
    return this;
  }
  public void unsetKey() {
    this.key = null;
  }
  public boolean isSetKey() {
    return this.key != null;
  }
  public void setKeyIsSet(boolean value) {
    if (!value) {
      this.key = null;
    }
  }
  public int getValuesSize() {
    return (this.values == null) ? 0 : this.values.size();
  }
  public java.util.Iterator<VersionedValue> getValuesIterator() {
    return (this.values == null) ? null : this.values.iterator();
  }
  public void addToValues(VersionedValue elem) {
    if (this.values == null) {
      this.values = new ArrayList<VersionedValue>();
    }
    this.values.add(elem);
  }
  public List<VersionedValue> getValues() {
    return this.values;
  }
  public KeyedValues setValues(List<VersionedValue> values) {
    this.values = values;
    return this;
  }
  public void unsetValues() {
    this.values = null;
  }
  public boolean isSetValues() {
    return this.values != null;
  }
  public void setValuesIsSet(boolean value) {
    if (!value) {
      this.values = null;
    }
  }
  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case KEY:
      if (value == null) {
        unsetKey();
      } else {
        setKey((ByteBuffer)value);
      }
      break;
    case VALUES:
      if (value == null) {
        unsetValues();
      } else {
        setValues((List<VersionedValue>)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case KEY:
      return getKey();
    case VALUES:
      return getValues();
    }
    throw new IllegalStateException();
  }
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }
    switch (field) {
    case KEY:
      return isSetKey();
    case VALUES:
      return isSetValues();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof KeyedValues)
      return this.equals((KeyedValues)that);
    return false;
  }
  public boolean equals(KeyedValues that) {
    if (that == null)
      return false;
    boolean this_present_key = true && this.isSetKey();
    boolean that_present_key = true && that.isSetKey();
    if (this_present_key || that_present_key) {
      if (!(this_present_key && that_present_key))
        return false;
      if (!this.key.equals(that.key))
        return false;
    }
    boolean this_present_values = true && this.isSetValues();
    boolean that_present_values = true && that.isSetValues();
    if (this_present_values || that_present_values) {
      if (!(this_present_values && that_present_values))
        return false;
      if (!this.values.equals(that.values))
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(KeyedValues other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    KeyedValues typedOther = (KeyedValues)other;
    lastComparison = Boolean.valueOf(isSetKey()).compareTo(typedOther.isSetKey());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetKey()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.key, typedOther.key);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetValues()).compareTo(typedOther.isSetValues());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetValues()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.values, typedOther.values);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    return 0;
  }
  public _Fields fieldForId(int fieldId) {
    return _Fields.findByThriftId(fieldId);
  }
  public void read(org.apache.thrift.protocol.TProtocol iprot) throws org.apache.thrift.TException {
    schemes.get(iprot.getScheme()).getScheme().read(iprot, this);
  }
  public void write(org.apache.thrift.protocol.TProtocol oprot) throws org.apache.thrift.TException {
    schemes.get(oprot.getScheme()).getScheme().write(oprot, this);
  }
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("KeyedValues(");
    boolean first = true;
    sb.append("key:");
    if (this.key == null) {
      sb.append("null");
    } else {
      org.apache.thrift.TBaseHelper.toString(this.key, sb);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("values:");
    if (this.values == null) {
      sb.append("null");
    } else {
      sb.append(this.values);
    }
    first = false;
    sb.append(")");
    return sb.toString();
  }
  public void validate() throws org.apache.thrift.TException {
    if (key == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'key' was not present! Struct: " + toString());
    }
    if (values == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'values' was not present! Struct: " + toString());
    }
  }
  private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
    try {
      write(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(out)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }
  private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
    try {
      read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }
  private static class KeyedValuesStandardSchemeFactory implements SchemeFactory {
    public KeyedValuesStandardScheme getScheme() {
      return new KeyedValuesStandardScheme();
    }
  }
  private static class KeyedValuesStandardScheme extends StandardScheme<KeyedValues> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, KeyedValues struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
            if (schemeField.type == org.apache.thrift.protocol.TType.STRING) {
              struct.key = iprot.readBinary();
              struct.setKeyIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list8 = iprot.readListBegin();
                struct.values = new ArrayList<VersionedValue>(_list8.size);
                for (int _i9 = 0; _i9 < _list8.size; ++_i9)
                {
                  _elem10 = new VersionedValue();
                  _elem10.read(iprot);
                  struct.values.add(_elem10);
                }
                iprot.readListEnd();
              }
              struct.setValuesIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          default:
            org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
        }
        iprot.readFieldEnd();
      }
      iprot.readStructEnd();
      struct.validate();
    }
    public void write(org.apache.thrift.protocol.TProtocol oprot, KeyedValues struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.key != null) {
        oprot.writeFieldBegin(KEY_FIELD_DESC);
        oprot.writeBinary(struct.key);
        oprot.writeFieldEnd();
      }
      if (struct.values != null) {
        oprot.writeFieldBegin(VALUES_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, struct.values.size()));
          for (VersionedValue _iter11 : struct.values)
          {
            _iter11.write(oprot);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class KeyedValuesTupleSchemeFactory implements SchemeFactory {
    public KeyedValuesTupleScheme getScheme() {
      return new KeyedValuesTupleScheme();
    }
  }
  private static class KeyedValuesTupleScheme extends TupleScheme<KeyedValues> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, KeyedValues struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      oprot.writeBinary(struct.key);
      {
        oprot.writeI32(struct.values.size());
        for (VersionedValue _iter12 : struct.values)
        {
          _iter12.write(oprot);
        }
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, KeyedValues struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      struct.key = iprot.readBinary();
      struct.setKeyIsSet(true);
      {
        org.apache.thrift.protocol.TList _list13 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, iprot.readI32());
        struct.values = new ArrayList<VersionedValue>(_list13.size);
        for (int _i14 = 0; _i14 < _list13.size; ++_i14)
        {
          _elem15 = new VersionedValue();
          _elem15.read(iprot);
          struct.values.add(_elem15);
        }
      }
      struct.setValuesIsSet(true);
    }
  }
}
