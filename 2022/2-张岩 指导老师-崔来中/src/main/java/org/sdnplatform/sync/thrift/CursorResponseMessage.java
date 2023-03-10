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
@SuppressWarnings("all") public class CursorResponseMessage implements org.apache.thrift.TBase<CursorResponseMessage, CursorResponseMessage._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("CursorResponseMessage");
  private static final org.apache.thrift.protocol.TField HEADER_FIELD_DESC = new org.apache.thrift.protocol.TField("header", org.apache.thrift.protocol.TType.STRUCT, (short)1);
  private static final org.apache.thrift.protocol.TField CURSOR_ID_FIELD_DESC = new org.apache.thrift.protocol.TField("cursorId", org.apache.thrift.protocol.TType.I32, (short)2);
  private static final org.apache.thrift.protocol.TField VALUES_FIELD_DESC = new org.apache.thrift.protocol.TField("values", org.apache.thrift.protocol.TType.LIST, (short)3);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new CursorResponseMessageStandardSchemeFactory());
    schemes.put(TupleScheme.class, new CursorResponseMessageTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    HEADER((short)1, "header"),
    CURSOR_ID((short)2, "cursorId"),
    VALUES((short)3, "values");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return HEADER;
          return CURSOR_ID;
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
  private static final int __CURSORID_ISSET_ID = 0;
  private byte __isset_bitfield = 0;
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.HEADER, new org.apache.thrift.meta_data.FieldMetaData("header", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, AsyncMessageHeader.class)));
    tmpMap.put(_Fields.CURSOR_ID, new org.apache.thrift.meta_data.FieldMetaData("cursorId", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32)));
    tmpMap.put(_Fields.VALUES, new org.apache.thrift.meta_data.FieldMetaData("values", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, KeyedValues.class))));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(CursorResponseMessage.class, metaDataMap);
  }
  public CursorResponseMessage() {
  }
  public CursorResponseMessage(
    AsyncMessageHeader header,
    int cursorId,
    List<KeyedValues> values)
  {
    this();
    this.header = header;
    this.cursorId = cursorId;
    setCursorIdIsSet(true);
    this.values = values;
  }
  public CursorResponseMessage(CursorResponseMessage other) {
    __isset_bitfield = other.__isset_bitfield;
    if (other.isSetHeader()) {
      this.header = new AsyncMessageHeader(other.header);
    }
    this.cursorId = other.cursorId;
    if (other.isSetValues()) {
      List<KeyedValues> __this__values = new ArrayList<KeyedValues>();
      for (KeyedValues other_element : other.values) {
        __this__values.add(new KeyedValues(other_element));
      }
      this.values = __this__values;
    }
  }
  public CursorResponseMessage deepCopy() {
    return new CursorResponseMessage(this);
  }
  @Override
  public void clear() {
    this.header = null;
    setCursorIdIsSet(false);
    this.cursorId = 0;
    this.values = null;
  }
  public AsyncMessageHeader getHeader() {
    return this.header;
  }
  public CursorResponseMessage setHeader(AsyncMessageHeader header) {
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
  public int getCursorId() {
    return this.cursorId;
  }
  public CursorResponseMessage setCursorId(int cursorId) {
    this.cursorId = cursorId;
    setCursorIdIsSet(true);
    return this;
  }
  public void unsetCursorId() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __CURSORID_ISSET_ID);
  }
  public boolean isSetCursorId() {
    return EncodingUtils.testBit(__isset_bitfield, __CURSORID_ISSET_ID);
  }
  public void setCursorIdIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __CURSORID_ISSET_ID, value);
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
  public CursorResponseMessage setValues(List<KeyedValues> values) {
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
    case HEADER:
      if (value == null) {
        unsetHeader();
      } else {
        setHeader((AsyncMessageHeader)value);
      }
      break;
    case CURSOR_ID:
      if (value == null) {
        unsetCursorId();
      } else {
        setCursorId((Integer)value);
      }
      break;
    case VALUES:
      if (value == null) {
        unsetValues();
      } else {
        setValues((List<KeyedValues>)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case HEADER:
      return getHeader();
    case CURSOR_ID:
      return Integer.valueOf(getCursorId());
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
    case HEADER:
      return isSetHeader();
    case CURSOR_ID:
      return isSetCursorId();
    case VALUES:
      return isSetValues();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof CursorResponseMessage)
      return this.equals((CursorResponseMessage)that);
    return false;
  }
  public boolean equals(CursorResponseMessage that) {
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
    boolean this_present_cursorId = true;
    boolean that_present_cursorId = true;
    if (this_present_cursorId || that_present_cursorId) {
      if (!(this_present_cursorId && that_present_cursorId))
        return false;
      if (this.cursorId != that.cursorId)
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
  public int compareTo(CursorResponseMessage other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    CursorResponseMessage typedOther = (CursorResponseMessage)other;
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
    lastComparison = Boolean.valueOf(isSetCursorId()).compareTo(typedOther.isSetCursorId());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetCursorId()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.cursorId, typedOther.cursorId);
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
    StringBuilder sb = new StringBuilder("CursorResponseMessage(");
    boolean first = true;
    sb.append("header:");
    if (this.header == null) {
      sb.append("null");
    } else {
      sb.append(this.header);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("cursorId:");
    sb.append(this.cursorId);
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
    if (header == null) {
      throw new org.apache.thrift.protocol.TProtocolException("Required field 'header' was not present! Struct: " + toString());
    }
    if (header != null) {
      header.validate();
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
  private static class CursorResponseMessageStandardSchemeFactory implements SchemeFactory {
    public CursorResponseMessageStandardScheme getScheme() {
      return new CursorResponseMessageStandardScheme();
    }
  }
  private static class CursorResponseMessageStandardScheme extends StandardScheme<CursorResponseMessage> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, CursorResponseMessage struct) throws org.apache.thrift.TException {
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
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.cursorId = iprot.readI32();
              struct.setCursorIdIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list56 = iprot.readListBegin();
                struct.values = new ArrayList<KeyedValues>(_list56.size);
                for (int _i57 = 0; _i57 < _list56.size; ++_i57)
                {
                  _elem58 = new KeyedValues();
                  _elem58.read(iprot);
                  struct.values.add(_elem58);
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
      if (!struct.isSetCursorId()) {
        throw new org.apache.thrift.protocol.TProtocolException("Required field 'cursorId' was not found in serialized data! Struct: " + toString());
      }
      struct.validate();
    }
    public void write(org.apache.thrift.protocol.TProtocol oprot, CursorResponseMessage struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.header != null) {
        oprot.writeFieldBegin(HEADER_FIELD_DESC);
        struct.header.write(oprot);
        oprot.writeFieldEnd();
      }
      oprot.writeFieldBegin(CURSOR_ID_FIELD_DESC);
      oprot.writeI32(struct.cursorId);
      oprot.writeFieldEnd();
      if (struct.values != null) {
        oprot.writeFieldBegin(VALUES_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, struct.values.size()));
          for (KeyedValues _iter59 : struct.values)
          {
            _iter59.write(oprot);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class CursorResponseMessageTupleSchemeFactory implements SchemeFactory {
    public CursorResponseMessageTupleScheme getScheme() {
      return new CursorResponseMessageTupleScheme();
    }
  }
  private static class CursorResponseMessageTupleScheme extends TupleScheme<CursorResponseMessage> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, CursorResponseMessage struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      struct.header.write(oprot);
      oprot.writeI32(struct.cursorId);
      BitSet optionals = new BitSet();
      if (struct.isSetValues()) {
        optionals.set(0);
      }
      oprot.writeBitSet(optionals, 1);
      if (struct.isSetValues()) {
        {
          oprot.writeI32(struct.values.size());
          for (KeyedValues _iter60 : struct.values)
          {
            _iter60.write(oprot);
          }
        }
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, CursorResponseMessage struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      struct.header = new AsyncMessageHeader();
      struct.header.read(iprot);
      struct.setHeaderIsSet(true);
      struct.cursorId = iprot.readI32();
      struct.setCursorIdIsSet(true);
      BitSet incoming = iprot.readBitSet(1);
      if (incoming.get(0)) {
        {
          org.apache.thrift.protocol.TList _list61 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, iprot.readI32());
          struct.values = new ArrayList<KeyedValues>(_list61.size);
          for (int _i62 = 0; _i62 < _list61.size; ++_i62)
          {
            _elem63 = new KeyedValues();
            _elem63.read(iprot);
            struct.values.add(_elem63);
          }
        }
        struct.setValuesIsSet(true);
      }
    }
  }
}
