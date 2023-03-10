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
@SuppressWarnings("all") public class HelloMessage implements org.apache.thrift.TBase<HelloMessage, HelloMessage._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("HelloMessage");
  private static final org.apache.thrift.protocol.TField HEADER_FIELD_DESC = new org.apache.thrift.protocol.TField("header", org.apache.thrift.protocol.TType.STRUCT, (short)1);
  private static final org.apache.thrift.protocol.TField NODE_ID_FIELD_DESC = new org.apache.thrift.protocol.TField("nodeId", org.apache.thrift.protocol.TType.I16, (short)2);
  private static final org.apache.thrift.protocol.TField AUTH_SCHEME_FIELD_DESC = new org.apache.thrift.protocol.TField("authScheme", org.apache.thrift.protocol.TType.I32, (short)3);
  private static final org.apache.thrift.protocol.TField AUTH_CHALLENGE_RESPONSE_FIELD_DESC = new org.apache.thrift.protocol.TField("authChallengeResponse", org.apache.thrift.protocol.TType.STRUCT, (short)4);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new HelloMessageStandardSchemeFactory());
    schemes.put(TupleScheme.class, new HelloMessageTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    HEADER((short)1, "header"),
    NODE_ID((short)2, "nodeId"),
    AUTH_SCHEME((short)3, "authScheme"),
    AUTH_CHALLENGE_RESPONSE((short)4, "authChallengeResponse");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return HEADER;
          return NODE_ID;
          return AUTH_SCHEME;
          return AUTH_CHALLENGE_RESPONSE;
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
  private static final int __NODEID_ISSET_ID = 0;
  private byte __isset_bitfield = 0;
  private _Fields optionals[] = {_Fields.NODE_ID,_Fields.AUTH_SCHEME,_Fields.AUTH_CHALLENGE_RESPONSE};
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.HEADER, new org.apache.thrift.meta_data.FieldMetaData("header", org.apache.thrift.TFieldRequirementType.REQUIRED, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, AsyncMessageHeader.class)));
    tmpMap.put(_Fields.NODE_ID, new org.apache.thrift.meta_data.FieldMetaData("nodeId", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I16)));
    tmpMap.put(_Fields.AUTH_SCHEME, new org.apache.thrift.meta_data.FieldMetaData("authScheme", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.EnumMetaData(org.apache.thrift.protocol.TType.ENUM, AuthScheme.class)));
    tmpMap.put(_Fields.AUTH_CHALLENGE_RESPONSE, new org.apache.thrift.meta_data.FieldMetaData("authChallengeResponse", org.apache.thrift.TFieldRequirementType.OPTIONAL, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, AuthChallengeResponse.class)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(HelloMessage.class, metaDataMap);
  }
  public HelloMessage() {
  }
  public HelloMessage(
    AsyncMessageHeader header)
  {
    this();
    this.header = header;
  }
  public HelloMessage(HelloMessage other) {
    __isset_bitfield = other.__isset_bitfield;
    if (other.isSetHeader()) {
      this.header = new AsyncMessageHeader(other.header);
    }
    this.nodeId = other.nodeId;
    if (other.isSetAuthScheme()) {
      this.authScheme = other.authScheme;
    }
    if (other.isSetAuthChallengeResponse()) {
      this.authChallengeResponse = new AuthChallengeResponse(other.authChallengeResponse);
    }
  }
  public HelloMessage deepCopy() {
    return new HelloMessage(this);
  }
  @Override
  public void clear() {
    this.header = null;
    setNodeIdIsSet(false);
    this.nodeId = 0;
    this.authScheme = null;
    this.authChallengeResponse = null;
  }
  public AsyncMessageHeader getHeader() {
    return this.header;
  }
  public HelloMessage setHeader(AsyncMessageHeader header) {
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
  public short getNodeId() {
    return this.nodeId;
  }
  public HelloMessage setNodeId(short nodeId) {
    this.nodeId = nodeId;
    setNodeIdIsSet(true);
    return this;
  }
  public void unsetNodeId() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __NODEID_ISSET_ID);
  }
  public boolean isSetNodeId() {
    return EncodingUtils.testBit(__isset_bitfield, __NODEID_ISSET_ID);
  }
  public void setNodeIdIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __NODEID_ISSET_ID, value);
  }
  public AuthScheme getAuthScheme() {
    return this.authScheme;
  }
  public HelloMessage setAuthScheme(AuthScheme authScheme) {
    this.authScheme = authScheme;
    return this;
  }
  public void unsetAuthScheme() {
    this.authScheme = null;
  }
  public boolean isSetAuthScheme() {
    return this.authScheme != null;
  }
  public void setAuthSchemeIsSet(boolean value) {
    if (!value) {
      this.authScheme = null;
    }
  }
  public AuthChallengeResponse getAuthChallengeResponse() {
    return this.authChallengeResponse;
  }
  public HelloMessage setAuthChallengeResponse(AuthChallengeResponse authChallengeResponse) {
    this.authChallengeResponse = authChallengeResponse;
    return this;
  }
  public void unsetAuthChallengeResponse() {
    this.authChallengeResponse = null;
  }
  public boolean isSetAuthChallengeResponse() {
    return this.authChallengeResponse != null;
  }
  public void setAuthChallengeResponseIsSet(boolean value) {
    if (!value) {
      this.authChallengeResponse = null;
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
    case NODE_ID:
      if (value == null) {
        unsetNodeId();
      } else {
        setNodeId((Short)value);
      }
      break;
    case AUTH_SCHEME:
      if (value == null) {
        unsetAuthScheme();
      } else {
        setAuthScheme((AuthScheme)value);
      }
      break;
    case AUTH_CHALLENGE_RESPONSE:
      if (value == null) {
        unsetAuthChallengeResponse();
      } else {
        setAuthChallengeResponse((AuthChallengeResponse)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case HEADER:
      return getHeader();
    case NODE_ID:
      return Short.valueOf(getNodeId());
    case AUTH_SCHEME:
      return getAuthScheme();
    case AUTH_CHALLENGE_RESPONSE:
      return getAuthChallengeResponse();
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
    case NODE_ID:
      return isSetNodeId();
    case AUTH_SCHEME:
      return isSetAuthScheme();
    case AUTH_CHALLENGE_RESPONSE:
      return isSetAuthChallengeResponse();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof HelloMessage)
      return this.equals((HelloMessage)that);
    return false;
  }
  public boolean equals(HelloMessage that) {
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
    boolean this_present_nodeId = true && this.isSetNodeId();
    boolean that_present_nodeId = true && that.isSetNodeId();
    if (this_present_nodeId || that_present_nodeId) {
      if (!(this_present_nodeId && that_present_nodeId))
        return false;
      if (this.nodeId != that.nodeId)
        return false;
    }
    boolean this_present_authScheme = true && this.isSetAuthScheme();
    boolean that_present_authScheme = true && that.isSetAuthScheme();
    if (this_present_authScheme || that_present_authScheme) {
      if (!(this_present_authScheme && that_present_authScheme))
        return false;
      if (!this.authScheme.equals(that.authScheme))
        return false;
    }
    boolean this_present_authChallengeResponse = true && this.isSetAuthChallengeResponse();
    boolean that_present_authChallengeResponse = true && that.isSetAuthChallengeResponse();
    if (this_present_authChallengeResponse || that_present_authChallengeResponse) {
      if (!(this_present_authChallengeResponse && that_present_authChallengeResponse))
        return false;
      if (!this.authChallengeResponse.equals(that.authChallengeResponse))
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(HelloMessage other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    HelloMessage typedOther = (HelloMessage)other;
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
    lastComparison = Boolean.valueOf(isSetNodeId()).compareTo(typedOther.isSetNodeId());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetNodeId()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.nodeId, typedOther.nodeId);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetAuthScheme()).compareTo(typedOther.isSetAuthScheme());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetAuthScheme()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.authScheme, typedOther.authScheme);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetAuthChallengeResponse()).compareTo(typedOther.isSetAuthChallengeResponse());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetAuthChallengeResponse()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.authChallengeResponse, typedOther.authChallengeResponse);
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
    StringBuilder sb = new StringBuilder("HelloMessage(");
    boolean first = true;
    sb.append("header:");
    if (this.header == null) {
      sb.append("null");
    } else {
      sb.append(this.header);
    }
    first = false;
    if (isSetNodeId()) {
      if (!first) sb.append(", ");
      sb.append("nodeId:");
      sb.append(this.nodeId);
      first = false;
    }
    if (isSetAuthScheme()) {
      if (!first) sb.append(", ");
      sb.append("authScheme:");
      if (this.authScheme == null) {
        sb.append("null");
      } else {
        sb.append(this.authScheme);
      }
      first = false;
    }
    if (isSetAuthChallengeResponse()) {
      if (!first) sb.append(", ");
      sb.append("authChallengeResponse:");
      if (this.authChallengeResponse == null) {
        sb.append("null");
      } else {
        sb.append(this.authChallengeResponse);
      }
      first = false;
    }
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
    if (authChallengeResponse != null) {
      authChallengeResponse.validate();
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
  private static class HelloMessageStandardSchemeFactory implements SchemeFactory {
    public HelloMessageStandardScheme getScheme() {
      return new HelloMessageStandardScheme();
    }
  }
  private static class HelloMessageStandardScheme extends StandardScheme<HelloMessage> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, HelloMessage struct) throws org.apache.thrift.TException {
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
            if (schemeField.type == org.apache.thrift.protocol.TType.I16) {
              struct.nodeId = iprot.readI16();
              struct.setNodeIdIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.authScheme = AuthScheme.findByValue(iprot.readI32());
              struct.setAuthSchemeIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.STRUCT) {
              struct.authChallengeResponse = new AuthChallengeResponse();
              struct.authChallengeResponse.read(iprot);
              struct.setAuthChallengeResponseIsSet(true);
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
    public void write(org.apache.thrift.protocol.TProtocol oprot, HelloMessage struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.header != null) {
        oprot.writeFieldBegin(HEADER_FIELD_DESC);
        struct.header.write(oprot);
        oprot.writeFieldEnd();
      }
      if (struct.isSetNodeId()) {
        oprot.writeFieldBegin(NODE_ID_FIELD_DESC);
        oprot.writeI16(struct.nodeId);
        oprot.writeFieldEnd();
      }
      if (struct.authScheme != null) {
        if (struct.isSetAuthScheme()) {
          oprot.writeFieldBegin(AUTH_SCHEME_FIELD_DESC);
          oprot.writeI32(struct.authScheme.getValue());
          oprot.writeFieldEnd();
        }
      }
      if (struct.authChallengeResponse != null) {
        if (struct.isSetAuthChallengeResponse()) {
          oprot.writeFieldBegin(AUTH_CHALLENGE_RESPONSE_FIELD_DESC);
          struct.authChallengeResponse.write(oprot);
          oprot.writeFieldEnd();
        }
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class HelloMessageTupleSchemeFactory implements SchemeFactory {
    public HelloMessageTupleScheme getScheme() {
      return new HelloMessageTupleScheme();
    }
  }
  private static class HelloMessageTupleScheme extends TupleScheme<HelloMessage> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, HelloMessage struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      struct.header.write(oprot);
      BitSet optionals = new BitSet();
      if (struct.isSetNodeId()) {
        optionals.set(0);
      }
      if (struct.isSetAuthScheme()) {
        optionals.set(1);
      }
      if (struct.isSetAuthChallengeResponse()) {
        optionals.set(2);
      }
      oprot.writeBitSet(optionals, 3);
      if (struct.isSetNodeId()) {
        oprot.writeI16(struct.nodeId);
      }
      if (struct.isSetAuthScheme()) {
        oprot.writeI32(struct.authScheme.getValue());
      }
      if (struct.isSetAuthChallengeResponse()) {
        struct.authChallengeResponse.write(oprot);
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, HelloMessage struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      struct.header = new AsyncMessageHeader();
      struct.header.read(iprot);
      struct.setHeaderIsSet(true);
      BitSet incoming = iprot.readBitSet(3);
      if (incoming.get(0)) {
        struct.nodeId = iprot.readI16();
        struct.setNodeIdIsSet(true);
      }
      if (incoming.get(1)) {
        struct.authScheme = AuthScheme.findByValue(iprot.readI32());
        struct.setAuthSchemeIsSet(true);
      }
      if (incoming.get(2)) {
        struct.authChallengeResponse = new AuthChallengeResponse();
        struct.authChallengeResponse.read(iprot);
        struct.setAuthChallengeResponseIsSet(true);
      }
    }
  }
}
