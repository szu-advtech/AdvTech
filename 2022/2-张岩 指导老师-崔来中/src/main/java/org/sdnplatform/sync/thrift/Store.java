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
@SuppressWarnings("all") public class Store implements org.apache.thrift.TBase<Store, Store._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("Store");
  private static final org.apache.thrift.protocol.TField STORE_NAME_FIELD_DESC = new org.apache.thrift.protocol.TField("storeName", org.apache.thrift.protocol.TType.STRING, (short)1);
  private static final org.apache.thrift.protocol.TField SCOPE_FIELD_DESC = new org.apache.thrift.protocol.TField("scope", org.apache.thrift.protocol.TType.I32, (short)2);
  private static final org.apache.thrift.protocol.TField PERSIST_FIELD_DESC = new org.apache.thrift.protocol.TField("persist", org.apache.thrift.protocol.TType.BOOL, (short)3);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new StoreStandardSchemeFactory());
    schemes.put(TupleScheme.class, new StoreTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    STORE_NAME((short)1, "storeName"),
    SCOPE((short)2, "scope"),
    PERSIST((short)3, "persist");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return STORE_NAME;
          return SCOPE;
          return PERSIST;
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
  private static final int __PERSIST_ISSET_ID = 0;
  private byte __isset_bitfield = 0;
  private _Fields optionals[] = {_Fields.SCOPE,_Fields.PERSIST};
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.STORE_NAME, new org.apache.thrift.meta_data.FieldMetaData("storeName", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING)));
    tmpMap.put(_Fields.SCOPE, new org.apache.thrift.meta_data.FieldMetaData("scope", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.EnumMetaData(org.apache.thrift.protocol.TType.ENUM, Scope.class)));
    tmpMap.put(_Fields.PERSIST, new org.apache.thrift.meta_data.FieldMetaData("persist", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.BOOL)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(Store.class, metaDataMap);
  }
  public Store() {
  }
  public Store(
    String storeName)
  {
    this();
    this.storeName = storeName;
  }
  public Store(Store other) {
    __isset_bitfield = other.__isset_bitfield;
    if (other.isSetStoreName()) {
      this.storeName = other.storeName;
    }
    if (other.isSetScope()) {
      this.scope = other.scope;
    }
    this.persist = other.persist;
  }
  public Store deepCopy() {
    return new Store(this);
  }
  @Override
  public void clear() {
    this.storeName = null;
    this.scope = null;
    setPersistIsSet(false);
    this.persist = false;
  }
  public String getStoreName() {
    return this.storeName;
  }
  public Store setStoreName(String storeName) {
    this.storeName = storeName;
    return this;
  }
  public void unsetStoreName() {
    this.storeName = null;
  }
  public boolean isSetStoreName() {
    return this.storeName != null;
  }
  public void setStoreNameIsSet(boolean value) {
    if (!value) {
      this.storeName = null;
    }
  }
  public Scope getScope() {
    return this.scope;
  }
  public Store setScope(Scope scope) {
    this.scope = scope;
    return this;
  }
  public void unsetScope() {
    this.scope = null;
  }
  public boolean isSetScope() {
    return this.scope != null;
  }
  public void setScopeIsSet(boolean value) {
    if (!value) {
      this.scope = null;
    }
  }
  public boolean isPersist() {
    return this.persist;
  }
  public Store setPersist(boolean persist) {
    this.persist = persist;
    setPersistIsSet(true);
    return this;
  }
  public void unsetPersist() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __PERSIST_ISSET_ID);
  }
  public boolean isSetPersist() {
    return EncodingUtils.testBit(__isset_bitfield, __PERSIST_ISSET_ID);
  }
  public void setPersistIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __PERSIST_ISSET_ID, value);
  }
  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case STORE_NAME:
      if (value == null) {
        unsetStoreName();
      } else {
        setStoreName((String)value);
      }
      break;
    case SCOPE:
      if (value == null) {
        unsetScope();
      } else {
        setScope((Scope)value);
      }
      break;
    case PERSIST:
      if (value == null) {
        unsetPersist();
      } else {
        setPersist((Boolean)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case STORE_NAME:
      return getStoreName();
    case SCOPE:
      return getScope();
    case PERSIST:
      return Boolean.valueOf(isPersist());
    }
    throw new IllegalStateException();
  }
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }
    switch (field) {
    case STORE_NAME:
      return isSetStoreName();
    case SCOPE:
      return isSetScope();
    case PERSIST:
      return isSetPersist();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof Store)
      return this.equals((Store)that);
    return false;
  }
  public boolean equals(Store that) {
    if (that == null)
      return false;
    boolean this_present_storeName = true && this.isSetStoreName();
    boolean that_present_storeName = true && that.isSetStoreName();
    if (this_present_storeName || that_present_storeName) {
      if (!(this_present_storeName && that_present_storeName))
        return false;
      if (!this.storeName.equals(that.storeName))
        return false;
    }
    boolean this_present_scope = true && this.isSetScope();
    boolean that_present_scope = true && that.isSetScope();
    if (this_present_scope || that_present_scope) {
      if (!(this_present_scope && that_present_scope))
        return false;
      if (!this.scope.equals(that.scope))
        return false;
    }
    boolean this_present_persist = true && this.isSetPersist();
    boolean that_present_persist = true && that.isSetPersist();
    if (this_present_persist || that_present_persist) {
      if (!(this_present_persist && that_present_persist))
        return false;
      if (this.persist != that.persist)
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(Store other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    Store typedOther = (Store)other;
    lastComparison = Boolean.valueOf(isSetStoreName()).compareTo(typedOther.isSetStoreName());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetStoreName()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.storeName, typedOther.storeName);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetScope()).compareTo(typedOther.isSetScope());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetScope()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.scope, typedOther.scope);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetPersist()).compareTo(typedOther.isSetPersist());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetPersist()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.persist, typedOther.persist);
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
    StringBuilder sb = new StringBuilder("Store(");
    boolean first = true;
    sb.append("storeName:");
    if (this.storeName == null) {
      sb.append("null");
    } else {
      sb.append(this.storeName);
    }
    first = false;
    if (isSetScope()) {
      if (!first) sb.append(", ");
      sb.append("scope:");
      if (this.scope == null) {
        sb.append("null");
      } else {
        sb.append(this.scope);
      }
      first = false;
    }
    if (isSetPersist()) {
      if (!first) sb.append(", ");
      sb.append("persist:");
      sb.append(this.persist);
      first = false;
    }
    sb.append(")");
    return sb.toString();
  }
  public void validate() throws org.apache.thrift.TException {
    if (storeName == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'storeName' was not present! Struct: " + toString());
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
      __isset_bitfield = 0;
      read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }
  private static class StoreStandardSchemeFactory implements SchemeFactory {
    public StoreStandardScheme getScheme() {
      return new StoreStandardScheme();
    }
  }
  private static class StoreStandardScheme extends StandardScheme<Store> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, Store struct) throws org.apache.thrift.TException {
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
              struct.storeName = iprot.readString();
              struct.setStoreNameIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.scope = Scope.findByValue(iprot.readI32());
              struct.setScopeIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.BOOL) {
              struct.persist = iprot.readBool();
              struct.setPersistIsSet(true);
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
    public void write(org.apache.thrift.protocol.TProtocol oprot, Store struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.storeName != null) {
        oprot.writeFieldBegin(STORE_NAME_FIELD_DESC);
        oprot.writeString(struct.storeName);
        oprot.writeFieldEnd();
      }
      if (struct.scope != null) {
        if (struct.isSetScope()) {
          oprot.writeFieldBegin(SCOPE_FIELD_DESC);
          oprot.writeI32(struct.scope.getValue());
          oprot.writeFieldEnd();
        }
      }
      if (struct.isSetPersist()) {
        oprot.writeFieldBegin(PERSIST_FIELD_DESC);
        oprot.writeBool(struct.persist);
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class StoreTupleSchemeFactory implements SchemeFactory {
    public StoreTupleScheme getScheme() {
      return new StoreTupleScheme();
    }
  }
  private static class StoreTupleScheme extends TupleScheme<Store> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, Store struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      oprot.writeString(struct.storeName);
      BitSet optionals = new BitSet();
      if (struct.isSetScope()) {
        optionals.set(0);
      }
      if (struct.isSetPersist()) {
        optionals.set(1);
      }
      oprot.writeBitSet(optionals, 2);
      if (struct.isSetScope()) {
        oprot.writeI32(struct.scope.getValue());
      }
      if (struct.isSetPersist()) {
        oprot.writeBool(struct.persist);
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, Store struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      struct.storeName = iprot.readString();
      struct.setStoreNameIsSet(true);
      BitSet incoming = iprot.readBitSet(2);
      if (incoming.get(0)) {
        struct.scope = Scope.findByValue(iprot.readI32());
        struct.setScopeIsSet(true);
      }
      if (incoming.get(1)) {
        struct.persist = iprot.readBool();
        struct.setPersistIsSet(true);
      }
    }
  }
}
