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
@SuppressWarnings("all") public class AsyncMessageHeader implements org.apache.thrift.TBase<AsyncMessageHeader, AsyncMessageHeader._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("AsyncMessageHeader");
  private static final org.apache.thrift.protocol.TField TRANSACTION_ID_FIELD_DESC = new org.apache.thrift.protocol.TField("transactionId", org.apache.thrift.protocol.TType.I32, (short)1);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new AsyncMessageHeaderStandardSchemeFactory());
    schemes.put(TupleScheme.class, new AsyncMessageHeaderTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    TRANSACTION_ID((short)1, "transactionId");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return TRANSACTION_ID;
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
  private static final int __TRANSACTIONID_ISSET_ID = 0;
  private byte __isset_bitfield = 0;
  private _Fields optionals[] = {_Fields.TRANSACTION_ID};
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.TRANSACTION_ID, new org.apache.thrift.meta_data.FieldMetaData("transactionId", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(AsyncMessageHeader.class, metaDataMap);
  }
  public AsyncMessageHeader() {
  }
  public AsyncMessageHeader(AsyncMessageHeader other) {
    __isset_bitfield = other.__isset_bitfield;
    this.transactionId = other.transactionId;
  }
  public AsyncMessageHeader deepCopy() {
    return new AsyncMessageHeader(this);
  }
  @Override
  public void clear() {
    setTransactionIdIsSet(false);
    this.transactionId = 0;
  }
  public int getTransactionId() {
    return this.transactionId;
  }
  public AsyncMessageHeader setTransactionId(int transactionId) {
    this.transactionId = transactionId;
    setTransactionIdIsSet(true);
    return this;
  }
  public void unsetTransactionId() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __TRANSACTIONID_ISSET_ID);
  }
  public boolean isSetTransactionId() {
    return EncodingUtils.testBit(__isset_bitfield, __TRANSACTIONID_ISSET_ID);
  }
  public void setTransactionIdIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __TRANSACTIONID_ISSET_ID, value);
  }
  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case TRANSACTION_ID:
      if (value == null) {
        unsetTransactionId();
      } else {
        setTransactionId((Integer)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case TRANSACTION_ID:
      return Integer.valueOf(getTransactionId());
    }
    throw new IllegalStateException();
  }
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }
    switch (field) {
    case TRANSACTION_ID:
      return isSetTransactionId();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof AsyncMessageHeader)
      return this.equals((AsyncMessageHeader)that);
    return false;
  }
  public boolean equals(AsyncMessageHeader that) {
    if (that == null)
      return false;
    boolean this_present_transactionId = true && this.isSetTransactionId();
    boolean that_present_transactionId = true && that.isSetTransactionId();
    if (this_present_transactionId || that_present_transactionId) {
      if (!(this_present_transactionId && that_present_transactionId))
        return false;
      if (this.transactionId != that.transactionId)
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(AsyncMessageHeader other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    AsyncMessageHeader typedOther = (AsyncMessageHeader)other;
    lastComparison = Boolean.valueOf(isSetTransactionId()).compareTo(typedOther.isSetTransactionId());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetTransactionId()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.transactionId, typedOther.transactionId);
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
    StringBuilder sb = new StringBuilder("AsyncMessageHeader(");
    boolean first = true;
    if (isSetTransactionId()) {
      sb.append("transactionId:");
      sb.append(this.transactionId);
      first = false;
    }
    sb.append(")");
    return sb.toString();
  }
  public void validate() throws org.apache.thrift.TException {
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
  private static class AsyncMessageHeaderStandardSchemeFactory implements SchemeFactory {
    public AsyncMessageHeaderStandardScheme getScheme() {
      return new AsyncMessageHeaderStandardScheme();
    }
  }
  private static class AsyncMessageHeaderStandardScheme extends StandardScheme<AsyncMessageHeader> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, AsyncMessageHeader struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.transactionId = iprot.readI32();
              struct.setTransactionIdIsSet(true);
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
    public void write(org.apache.thrift.protocol.TProtocol oprot, AsyncMessageHeader struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.isSetTransactionId()) {
        oprot.writeFieldBegin(TRANSACTION_ID_FIELD_DESC);
        oprot.writeI32(struct.transactionId);
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class AsyncMessageHeaderTupleSchemeFactory implements SchemeFactory {
    public AsyncMessageHeaderTupleScheme getScheme() {
      return new AsyncMessageHeaderTupleScheme();
    }
  }
  private static class AsyncMessageHeaderTupleScheme extends TupleScheme<AsyncMessageHeader> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, AsyncMessageHeader struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      BitSet optionals = new BitSet();
      if (struct.isSetTransactionId()) {
        optionals.set(0);
      }
      oprot.writeBitSet(optionals, 1);
      if (struct.isSetTransactionId()) {
        oprot.writeI32(struct.transactionId);
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, AsyncMessageHeader struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      BitSet incoming = iprot.readBitSet(1);
      if (incoming.get(0)) {
        struct.transactionId = iprot.readI32();
        struct.setTransactionIdIsSet(true);
      }
    }
  }
}
