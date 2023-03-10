package net.floodlightcontroller.packetstreamer.thrift;
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
@SuppressWarnings("all") public class Message implements org.apache.thrift.TBase<Message, Message._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("Message");
  private static final org.apache.thrift.protocol.TField SESSION_IDS_FIELD_DESC = new org.apache.thrift.protocol.TField("sessionIDs", org.apache.thrift.protocol.TType.LIST, (short)1);
  private static final org.apache.thrift.protocol.TField PACKET_FIELD_DESC = new org.apache.thrift.protocol.TField("packet", org.apache.thrift.protocol.TType.STRUCT, (short)2);
  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new MessageStandardSchemeFactory());
    schemes.put(TupleScheme.class, new MessageTupleSchemeFactory());
  }
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    SESSION_IDS((short)1, "sessionIDs"),
    PACKET((short)2, "packet");
    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
          return SESSION_IDS;
          return PACKET;
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
    tmpMap.put(_Fields.SESSION_IDS, new org.apache.thrift.meta_data.FieldMetaData("sessionIDs", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING))));
    tmpMap.put(_Fields.PACKET, new org.apache.thrift.meta_data.FieldMetaData("packet", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, Packet.class)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(Message.class, metaDataMap);
  }
  public Message() {
  }
  public Message(
    List<String> sessionIDs,
    Packet packet)
  {
    this();
    this.sessionIDs = sessionIDs;
    this.packet = packet;
  }
  public Message(Message other) {
    if (other.isSetSessionIDs()) {
      List<String> __this__sessionIDs = new ArrayList<String>();
      for (String other_element : other.sessionIDs) {
        __this__sessionIDs.add(other_element);
      }
      this.sessionIDs = __this__sessionIDs;
    }
    if (other.isSetPacket()) {
      this.packet = new Packet(other.packet);
    }
  }
  public Message deepCopy() {
    return new Message(this);
  }
  @Override
  public void clear() {
    this.sessionIDs = null;
    this.packet = null;
  }
  public int getSessionIDsSize() {
    return (this.sessionIDs == null) ? 0 : this.sessionIDs.size();
  }
  public java.util.Iterator<String> getSessionIDsIterator() {
    return (this.sessionIDs == null) ? null : this.sessionIDs.iterator();
  }
  public void addToSessionIDs(String elem) {
    if (this.sessionIDs == null) {
      this.sessionIDs = new ArrayList<String>();
    }
    this.sessionIDs.add(elem);
  }
  public List<String> getSessionIDs() {
    return this.sessionIDs;
  }
  public Message setSessionIDs(List<String> sessionIDs) {
    this.sessionIDs = sessionIDs;
    return this;
  }
  public void unsetSessionIDs() {
    this.sessionIDs = null;
  }
  public boolean isSetSessionIDs() {
    return this.sessionIDs != null;
  }
  public void setSessionIDsIsSet(boolean value) {
    if (!value) {
      this.sessionIDs = null;
    }
  }
  public Packet getPacket() {
    return this.packet;
  }
  public Message setPacket(Packet packet) {
    this.packet = packet;
    return this;
  }
  public void unsetPacket() {
    this.packet = null;
  }
  public boolean isSetPacket() {
    return this.packet != null;
  }
  public void setPacketIsSet(boolean value) {
    if (!value) {
      this.packet = null;
    }
  }
  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case SESSION_IDS:
      if (value == null) {
        unsetSessionIDs();
      } else {
        setSessionIDs((List<String>)value);
      }
      break;
    case PACKET:
      if (value == null) {
        unsetPacket();
      } else {
        setPacket((Packet)value);
      }
      break;
    }
  }
  public Object getFieldValue(_Fields field) {
    switch (field) {
    case SESSION_IDS:
      return getSessionIDs();
    case PACKET:
      return getPacket();
    }
    throw new IllegalStateException();
  }
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }
    switch (field) {
    case SESSION_IDS:
      return isSetSessionIDs();
    case PACKET:
      return isSetPacket();
    }
    throw new IllegalStateException();
  }
  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof Message)
      return this.equals((Message)that);
    return false;
  }
  public boolean equals(Message that) {
    if (that == null)
      return false;
    boolean this_present_sessionIDs = true && this.isSetSessionIDs();
    boolean that_present_sessionIDs = true && that.isSetSessionIDs();
    if (this_present_sessionIDs || that_present_sessionIDs) {
      if (!(this_present_sessionIDs && that_present_sessionIDs))
        return false;
      if (!this.sessionIDs.equals(that.sessionIDs))
        return false;
    }
    boolean this_present_packet = true && this.isSetPacket();
    boolean that_present_packet = true && that.isSetPacket();
    if (this_present_packet || that_present_packet) {
      if (!(this_present_packet && that_present_packet))
        return false;
      if (!this.packet.equals(that.packet))
        return false;
    }
    return true;
  }
  @Override
  public int hashCode() {
    return 0;
  }
  public int compareTo(Message other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }
    int lastComparison = 0;
    Message typedOther = (Message)other;
    lastComparison = Boolean.valueOf(isSetSessionIDs()).compareTo(typedOther.isSetSessionIDs());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetSessionIDs()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.sessionIDs, typedOther.sessionIDs);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetPacket()).compareTo(typedOther.isSetPacket());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetPacket()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.packet, typedOther.packet);
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
    StringBuilder sb = new StringBuilder("Message(");
    boolean first = true;
    sb.append("sessionIDs:");
    if (this.sessionIDs == null) {
      sb.append("null");
    } else {
      sb.append(this.sessionIDs);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("packet:");
    if (this.packet == null) {
      sb.append("null");
    } else {
      sb.append(this.packet);
    }
    first = false;
    sb.append(")");
    return sb.toString();
  }
  public void validate() throws org.apache.thrift.TException {
    if (packet != null) {
      packet.validate();
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
  private static class MessageStandardSchemeFactory implements SchemeFactory {
    public MessageStandardScheme getScheme() {
      return new MessageStandardScheme();
    }
  }
  private static class MessageStandardScheme extends StandardScheme<Message> {
    public void read(org.apache.thrift.protocol.TProtocol iprot, Message struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list0 = iprot.readListBegin();
                struct.sessionIDs = new ArrayList<String>(_list0.size);
                for (int _i1 = 0; _i1 < _list0.size; ++_i1)
                {
                  _elem2 = iprot.readString();
                  struct.sessionIDs.add(_elem2);
                }
                iprot.readListEnd();
              }
              struct.setSessionIDsIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
            if (schemeField.type == org.apache.thrift.protocol.TType.STRUCT) {
              struct.packet = new Packet();
              struct.packet.read(iprot);
              struct.setPacketIsSet(true);
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
    public void write(org.apache.thrift.protocol.TProtocol oprot, Message struct) throws org.apache.thrift.TException {
      struct.validate();
      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.sessionIDs != null) {
        oprot.writeFieldBegin(SESSION_IDS_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRING, struct.sessionIDs.size()));
          for (String _iter3 : struct.sessionIDs)
          {
            oprot.writeString(_iter3);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      if (struct.packet != null) {
        oprot.writeFieldBegin(PACKET_FIELD_DESC);
        struct.packet.write(oprot);
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }
  }
  private static class MessageTupleSchemeFactory implements SchemeFactory {
    public MessageTupleScheme getScheme() {
      return new MessageTupleScheme();
    }
  }
  private static class MessageTupleScheme extends TupleScheme<Message> {
    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, Message struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      BitSet optionals = new BitSet();
      if (struct.isSetSessionIDs()) {
        optionals.set(0);
      }
      if (struct.isSetPacket()) {
        optionals.set(1);
      }
      oprot.writeBitSet(optionals, 2);
      if (struct.isSetSessionIDs()) {
        {
          oprot.writeI32(struct.sessionIDs.size());
          for (String _iter4 : struct.sessionIDs)
          {
            oprot.writeString(_iter4);
          }
        }
      }
      if (struct.isSetPacket()) {
        struct.packet.write(oprot);
      }
    }
    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, Message struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      BitSet incoming = iprot.readBitSet(2);
      if (incoming.get(0)) {
        {
          org.apache.thrift.protocol.TList _list5 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRING, iprot.readI32());
          struct.sessionIDs = new ArrayList<String>(_list5.size);
          for (int _i6 = 0; _i6 < _list5.size; ++_i6)
          {
            _elem7 = iprot.readString();
            struct.sessionIDs.add(_elem7);
          }
        }
        struct.setSessionIDsIsSet(true);
      }
      if (incoming.get(1)) {
        struct.packet = new Packet();
        struct.packet.read(iprot);
        struct.setPacketIsSet(true);
      }
    }
  }
}
