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
@SuppressWarnings("all") public class SyncValueMessage implements org.apache.thrift.TBase<SyncValueMessage, SyncValueMessage._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("SyncValueMessage");
  private static final org.apache.thrift.protocol.TField HEADER_FIELD_DESC = new org.apache.thrift.protocol.TField("header", org.apache.thrift.protocol.TType.STRUCT, (short)1);
  private static final org.apache.thrift.protocol.TField STORE_FIELD_DESC = new org.apache.thrift.protocol.TField("store", org.apache.thrift.protocol.TType.STRUCT, (short)2);
  private static final org.apache.thrift.protocol.TField VALUES_FIELD_DESC = new org.apache.thrift.protocol.TField("values", org.apache.thrift.protocol.TType.LIST, (short)3);
  private static final org.apache.thrift.protocol.TField RESPONSE_TO_FIELD_DESC = new org.apache.thrift.protocol.TField("responseTo", org.apache.thrift.protocol.TType.I32, (short)4);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new SyncValueMessageStandardSchemeFactory());
    schemes.put(TupleScheme.class, new SyncValueMessageTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    HEADER((short)1, "header"),
    STORE((short)2, "store"),
    VALUES((short)3, "values"),
    RESPONSE_TO((short)4, "responseTo");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return HEADER;
          return STORE;
          return VALUES;
          return RESPONSE_TO;
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
  private static final int __RESPONSETO_ISSET_ID = 0;
  private byte __isset_bitfield = 0;
  private _Fields optionals[] = {_Fields.RESPONSE_TO};
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.HEADER, new org.apache.thrift.meta_data.FieldMetaData("header", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, AsyncMessageHeader.class)));
    tmpMap.put(_Fields.STORE, new org.apache.thrift.meta_data.FieldMetaData("store", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, Store.class)));
    tmpMap.put(_Fields.VALUES, new org.apache.thrift.meta_data.FieldMetaData("values", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, KeyedValues.class))));
    tmpMap.put(_Fields.RESPONSE_TO, new org.apache.thrift.meta_data.FieldMetaData("responseTo", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(SyncValueMessage.class, metaDataMap);
  }
  public SyncValueMessage() {
  }
  public SyncValueMessage(
    AsyncMessageHeader header,
    Store store,
    List<KeyedValues> values)
  {
    this();
    this.header = header;
    this.store = store;
    this.values = values;
  }
  public SyncValueMessage(SyncValueMessage other) {
    __isset_bitfield = other.__isset_bitfield;
    if (other.isSetHeader()) {
      this.header = new AsyncMessageHeader(other.header);
    }
    if (other.isSetStore()) {
      this.store = new Store(other.store);
    }
    if (other.isSetValues()) {
      List<KeyedValues> __this__values = new ArrayList<KeyedValues>();
      for (KeyedValues other_element : other.values) {
        __this__values.add(new KeyedValues(other_element));
      }
      this.values = __this__values;
    }
    this.responseTo = other.responseTo;
  }
  public SyncValueMessage deepCopy() {
    return new SyncValueMessage(this);
  }
  @Override
  public void clear() {
    this.header = null;
    this.store = null;
    this.values = null;
    setResponseToIsSet(false);
    this.responseTo = 0;
  }
  public AsyncMessageHeader getHeader() {
    return this.header;
  }
  public SyncValueMessage setHeader(AsyncMessageHeader header) {
    this.header = header;
    return this;
  }
  public void unsetHeader() {
    this.header = null;
  }
  public boolean isSetHeader() {
    return this.header != null;
  }
  public void setHeaderIsSet(boolean value) {
    if (!value) {
      this.header = null;
    }
  }
  public Store getStore() {
    return this.store;
  }
  public SyncValueMessage setStore(Store store) {
    this.store = store;
    return this;
  }
  public void unsetStore() {
    this.store = null;
  }
  public boolean isSetStore() {
    return this.store != null;
  }
  public void setStoreIsSet(boolean value) {
    if (!value) {
      this.store = null;
    }
  }
  public int getValuesSize() {
    return (this.values == null) ? 0 : this.values.size();
  }
  public java.util.Iterator<KeyedValues> getValuesIterator() {
    return (this.values == null) ? null : this.values.iterator();
  }
  public void addToValues(KeyedValues elem) {
    if (this.values == null) {
      this.values = new ArrayList<KeyedValues>();
    }
    this.values.add(elem);
  }
  public List<KeyedValues> getValues() {
    return this.values;
  }
  public SyncValueMessage setValues(List<KeyedValues> values) {
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
  public int getResponseTo() {
    return this.responseTo;
  }
  public SyncValueMessage setResponseTo(int responseTo) {
    this.responseTo = responseTo;
    setResponseToIsSet(true);
    return this;
  }
  public void unsetResponseTo() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __RESPONSETO_ISSET_ID);
  }
  public boolean isSetResponseTo() {
    return EncodingUtils.testBit(__isset_bitfield, __RESPONSETO_ISSET_ID);
  }
  public void setResponseToIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __RESPONSETO_ISSET_ID, value);
  }
  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case HEADER:
      if (value == null) {
        unsetHeader();
      } else {
        setHeader((AsyncMessageHeader)value);
      }
      break;
    case STORE:
      if (value == null) {
        unsetStore();
      } else {
        setStore((Store)value);
      }
      break;
    case VALUES:
      if (value == null) {
        unsetValues();
      } else {
        setValues((List<KeyedValues>)value);
      }
      break;
    case RESPONSE_TO:
      if (value == null) {
        unsetResponseTo();
      } else {
        setResponseTo((Integer)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case HEADER:
      return getHeader();
    case STORE:
      return getStore();
    case VALUES:
      return getValues();
    case RESPONSE_TO:
      return Integer.valueOf(getResponseTo());
    }
    throw new IllegalStateException();
  }
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }
    switch (field) {
    case HEADER:
      return isSetHeader();
    case STORE:
      return isSetStore();
    case VALUES:
      return isSetValues();
    case RESPONSE_TO:
      return isSetResponseTo();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof SyncValueMessage)
      return this.equals((SyncValueMessage)that);
    return false;
  }
  public boolean equals(SyncValueMessage that) {
    if (that == null)
      return false;
    boolean this_present_header = true && this.isSetHeader();
    boolean that_present_header = true && that.isSetHeader();
    if (this_present_header || that_present_header) {
      if (!(this_present_header && that_present_header))
        return false;
      if (!this.header.equals(that.header))
        return false;
    }
    boolean this_present_store = true && this.isSetStore();
    boolean that_present_store = true && that.isSetStore();
    if (this_present_store || that_present_store) {
      if (!(this_present_store && that_present_store))
        return false;
      if (!this.store.equals(that.store))
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
    boolean this_present_responseTo = true && this.isSetResponseTo();
    boolean that_present_responseTo = true && that.isSetResponseTo();
    if (this_present_responseTo || that_present_responseTo) {
      if (!(this_present_responseTo && that_present_responseTo))
        return false;
      if (this.responseTo != that.responseTo)
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(SyncValueMessage other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    SyncValueMessage typedOther = (SyncValueMessage)other;
    lastComparison = Boolean.valueOf(isSetHeader()).compareTo(typedOther.isSetHeader());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetHeader()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.header, typedOther.header);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetStore()).compareTo(typedOther.isSetStore());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetStore()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.store, typedOther.store);
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
    lastComparison = Boolean.valueOf(isSetResponseTo()).compareTo(typedOther.isSetResponseTo());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetResponseTo()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.responseTo, typedOther.responseTo);
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
    StringBuilder sb = new StringBuilder("SyncValueMessage(");
    boolean first = true;
    sb.append("header:");
    if (this.header == null) {
      sb.append("null");
    } else {
      sb.append(this.header);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("store:");
    if (this.store == null) {
      sb.append("null");
    } else {
      sb.append(this.store);
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
    if (isSetResponseTo()) {
      if (!first) sb.append(", ");
      sb.append("responseTo:");
      sb.append(this.responseTo);
      first = false;
    }
    sb.append(")");
    return sb.toString();
  }
  public void validate() throws org.apache.thrift.TException {
    if (header == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'header' was not present! Struct: " + toString());
    }
    if (store == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'store' was not present! Struct: " + toString());
    }
    if (header != null) {
      header.validate();
    }
    if (store != null) {
      store.validate();
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
  private static class SyncValueMessageStandardSchemeFactory implements SchemeFactory {
    public SyncValueMessageStandardScheme getScheme() {
      return new SyncValueMessageStandardScheme();
    }
  }
  private static class SyncValueMessageStandardScheme extends StandardScheme<SyncValueMessage> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, SyncValueMessage struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
            if (schemeField.type == org.apache.thrift.protocol.TType.STRUCT) {
              struct.header = new AsyncMessageHeader();
              struct.header.read(iprot);
              struct.setHeaderIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.STRUCT) {
              struct.store = new Store();
              struct.store.read(iprot);
              struct.setStoreIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list32 = iprot.readListBegin();
                struct.values = new ArrayList<KeyedValues>(_list32.size);
                for (int _i33 = 0; _i33 < _list32.size; ++_i33)
                {
                  _elem34 = new KeyedValues();
                  _elem34.read(iprot);
                  struct.values.add(_elem34);
                }
                iprot.readListEnd();
              }
              struct.setValuesIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.responseTo = iprot.readI32();
              struct.setResponseToIsSet(true);
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
    public void write(org.apache.thrift.protocol.TProtocol oprot, SyncValueMessage struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.header != null) {
        oprot.writeFieldBegin(HEADER_FIELD_DESC);
        struct.header.write(oprot);
        oprot.writeFieldEnd();
      }
      if (struct.store != null) {
        oprot.writeFieldBegin(STORE_FIELD_DESC);
        struct.store.write(oprot);
        oprot.writeFieldEnd();
      }
      if (struct.values != null) {
        oprot.writeFieldBegin(VALUES_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, struct.values.size()));
          for (KeyedValues _iter35 : struct.values)
          {
            _iter35.write(oprot);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      if (struct.isSetResponseTo()) {
        oprot.writeFieldBegin(RESPONSE_TO_FIELD_DESC);
        oprot.writeI32(struct.responseTo);
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class SyncValueMessageTupleSchemeFactory implements SchemeFactory {
    public SyncValueMessageTupleScheme getScheme() {
      return new SyncValueMessageTupleScheme();
    }
  }
  private static class SyncValueMessageTupleScheme extends TupleScheme<SyncValueMessage> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, SyncValueMessage struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      struct.header.write(oprot);
      struct.store.write(oprot);
      BitSet optionals = new BitSet();
      if (struct.isSetValues()) {
        optionals.set(0);
      }
      if (struct.isSetResponseTo()) {
        optionals.set(1);
      }
      oprot.writeBitSet(optionals, 2);
      if (struct.isSetValues()) {
        {
          oprot.writeI32(struct.values.size());
          for (KeyedValues _iter36 : struct.values)
          {
            _iter36.write(oprot);
          }
        }
      }
      if (struct.isSetResponseTo()) {
        oprot.writeI32(struct.responseTo);
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, SyncValueMessage struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      struct.header = new AsyncMessageHeader();
      struct.header.read(iprot);
      struct.setHeaderIsSet(true);
      struct.store = new Store();
      struct.store.read(iprot);
      struct.setStoreIsSet(true);
      BitSet incoming = iprot.readBitSet(2);
      if (incoming.get(0)) {
        {
          org.apache.thrift.protocol.TList _list37 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, iprot.readI32());
          struct.values = new ArrayList<KeyedValues>(_list37.size);
          for (int _i38 = 0; _i38 < _list37.size; ++_i38)
          {
            _elem39 = new KeyedValues();
            _elem39.read(iprot);
            struct.values.add(_elem39);
          }
        }
        struct.setValuesIsSet(true);
      }
      if (incoming.get(1)) {
        struct.responseTo = iprot.readI32();
        struct.setResponseToIsSet(true);
      }
    }
  }
}
