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
@SuppressWarnings("all") public class VersionedValue implements org.apache.thrift.TBase<VersionedValue, VersionedValue._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("VersionedValue");
  private static final org.apache.thrift.protocol.TField VALUE_FIELD_DESC = new org.apache.thrift.protocol.TField("value", org.apache.thrift.protocol.TType.STRING, (short)1);
  private static final org.apache.thrift.protocol.TField VERSION_FIELD_DESC = new org.apache.thrift.protocol.TField("version", org.apache.thrift.protocol.TType.STRUCT, (short)2);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new VersionedValueStandardSchemeFactory());
    schemes.put(TupleScheme.class, new VersionedValueTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    VALUE((short)1, "value"),
    VERSION((short)2, "version");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return VALUE;
          return VERSION;
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
  private _Fields optionals[] = {_Fields.VALUE};
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.VALUE, new org.apache.thrift.meta_data.FieldMetaData("value", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING        , true)));
    tmpMap.put(_Fields.VERSION, new org.apache.thrift.meta_data.FieldMetaData("version", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, VectorClock.class)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(VersionedValue.class, metaDataMap);
  }
  public VersionedValue() {
  }
  public VersionedValue(
    VectorClock version)
  {
    this();
    this.version = version;
  }
  public VersionedValue(VersionedValue other) {
    if (other.isSetValue()) {
      this.value = org.apache.thrift.TBaseHelper.copyBinary(other.value);
;
    }
    if (other.isSetVersion()) {
      this.version = new VectorClock(other.version);
    }
  }
  public VersionedValue deepCopy() {
    return new VersionedValue(this);
  }
  @Override
  public void clear() {
    this.value = null;
    this.version = null;
  }
  public byte[] getValue() {
    setValue(org.apache.thrift.TBaseHelper.rightSize(value));
    return value == null ? null : value.array();
  }
  public ByteBuffer bufferForValue() {
    return value;
  }
  public VersionedValue setValue(byte[] value) {
    setValue(value == null ? (ByteBuffer)null : ByteBuffer.wrap(value));
    return this;
  }
  public VersionedValue setValue(ByteBuffer value) {
    this.value = value;
    return this;
  }
  public void unsetValue() {
    this.value = null;
  }
  public boolean isSetValue() {
    return this.value != null;
  }
  public void setValueIsSet(boolean value) {
    if (!value) {
      this.value = null;
    }
  }
  public VectorClock getVersion() {
    return this.version;
  }
  public VersionedValue setVersion(VectorClock version) {
    this.version = version;
    return this;
  }
  public void unsetVersion() {
    this.version = null;
  }
  public boolean isSetVersion() {
    return this.version != null;
  }
  public void setVersionIsSet(boolean value) {
    if (!value) {
      this.version = null;
    }
  }
  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case VALUE:
      if (value == null) {
        unsetValue();
      } else {
        setValue((ByteBuffer)value);
      }
      break;
    case VERSION:
      if (value == null) {
        unsetVersion();
      } else {
        setVersion((VectorClock)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case VALUE:
      return getValue();
    case VERSION:
      return getVersion();
    }
    throw new IllegalStateException();
  }
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }
    switch (field) {
    case VALUE:
      return isSetValue();
    case VERSION:
      return isSetVersion();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof VersionedValue)
      return this.equals((VersionedValue)that);
    return false;
  }
  public boolean equals(VersionedValue that) {
    if (that == null)
      return false;
    boolean this_present_value = true && this.isSetValue();
    boolean that_present_value = true && that.isSetValue();
    if (this_present_value || that_present_value) {
      if (!(this_present_value && that_present_value))
        return false;
      if (!this.value.equals(that.value))
        return false;
    }
    boolean this_present_version = true && this.isSetVersion();
    boolean that_present_version = true && that.isSetVersion();
    if (this_present_version || that_present_version) {
      if (!(this_present_version && that_present_version))
        return false;
      if (!this.version.equals(that.version))
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(VersionedValue other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    VersionedValue typedOther = (VersionedValue)other;
    lastComparison = Boolean.valueOf(isSetValue()).compareTo(typedOther.isSetValue());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetValue()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.value, typedOther.value);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetVersion()).compareTo(typedOther.isSetVersion());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetVersion()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.version, typedOther.version);
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
    StringBuilder sb = new StringBuilder("VersionedValue(");
    boolean first = true;
    if (isSetValue()) {
      sb.append("value:");
      if (this.value == null) {
        sb.append("null");
      } else {
        org.apache.thrift.TBaseHelper.toString(this.value, sb);
      }
      first = false;
    }
    if (!first) sb.append(", ");
    sb.append("version:");
    if (this.version == null) {
      sb.append("null");
    } else {
      sb.append(this.version);
    }
    first = false;
    sb.append(")");
    return sb.toString();
  }
  public void validate() throws org.apache.thrift.TException {
    if (version == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'version' was not present! Struct: " + toString());
    }
    if (version != null) {
      version.validate();
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
  private static class VersionedValueStandardSchemeFactory implements SchemeFactory {
    public VersionedValueStandardScheme getScheme() {
      return new VersionedValueStandardScheme();
    }
  }
  private static class VersionedValueStandardScheme extends StandardScheme<VersionedValue> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, VersionedValue struct) throws org.apache.thrift.TException {
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
              struct.value = iprot.readBinary();
              struct.setValueIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.STRUCT) {
              struct.version = new VectorClock();
              struct.version.read(iprot);
              struct.setVersionIsSet(true);
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
    public void write(org.apache.thrift.protocol.TProtocol oprot, VersionedValue struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.value != null) {
        if (struct.isSetValue()) {
          oprot.writeFieldBegin(VALUE_FIELD_DESC);
          oprot.writeBinary(struct.value);
          oprot.writeFieldEnd();
        }
      }
      if (struct.version != null) {
        oprot.writeFieldBegin(VERSION_FIELD_DESC);
        struct.version.write(oprot);
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class VersionedValueTupleSchemeFactory implements SchemeFactory {
    public VersionedValueTupleScheme getScheme() {
      return new VersionedValueTupleScheme();
    }
  }
  private static class VersionedValueTupleScheme extends TupleScheme<VersionedValue> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, VersionedValue struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      struct.version.write(oprot);
      BitSet optionals = new BitSet();
      if (struct.isSetValue()) {
        optionals.set(0);
      }
      oprot.writeBitSet(optionals, 1);
      if (struct.isSetValue()) {
        oprot.writeBinary(struct.value);
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, VersionedValue struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      struct.version = new VectorClock();
      struct.version.read(iprot);
      struct.setVersionIsSet(true);
      BitSet incoming = iprot.readBitSet(1);
      if (incoming.get(0)) {
        struct.value = iprot.readBinary();
        struct.setValueIsSet(true);
      }
    }
  }
}
