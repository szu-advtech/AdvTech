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
@SuppressWarnings("all") public class PacketStreamer {
  public interface Iface {
    public List<ByteBuffer> getPackets(String sessionid) throws org.apache.thrift.TException;
    public int pushMessageSync(Message packet) throws org.apache.thrift.TException;
    public void pushMessageAsync(Message packet) throws org.apache.thrift.TException;
    public void terminateSession(String sessionid) throws org.apache.thrift.TException;
  }
  public interface AsyncIface {
    public void getPackets(String sessionid, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.getPackets_call> resultHandler) throws org.apache.thrift.TException;
    public void pushMessageSync(Message packet, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.pushMessageSync_call> resultHandler) throws org.apache.thrift.TException;
    public void pushMessageAsync(Message packet, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.pushMessageAsync_call> resultHandler) throws org.apache.thrift.TException;
    public void terminateSession(String sessionid, org.apache.thrift.async.AsyncMethodCallback<AsyncClient.terminateSession_call> resultHandler) throws org.apache.thrift.TException;
  }
  public static class Client extends org.apache.thrift.TServiceClient implements Iface {
    public static class Factory implements org.apache.thrift.TServiceClientFactory<Client> {
      public Factory() {}
      public Client getClient(org.apache.thrift.protocol.TProtocol prot) {
        return new Client(prot);
      }
      public Client getClient(org.apache.thrift.protocol.TProtocol iprot, org.apache.thrift.protocol.TProtocol oprot) {
        return new Client(iprot, oprot);
      }
    }
    public Client(org.apache.thrift.protocol.TProtocol prot)
    {
      super(prot, prot);
    }
    public Client(org.apache.thrift.protocol.TProtocol iprot, org.apache.thrift.protocol.TProtocol oprot) {
      super(iprot, oprot);
    }
    public List<ByteBuffer> getPackets(String sessionid) throws org.apache.thrift.TException
    {
      send_getPackets(sessionid);
      return recv_getPackets();
    }
    public void send_getPackets(String sessionid) throws org.apache.thrift.TException
    {
      getPackets_args args = new getPackets_args();
      args.setSessionid(sessionid);
      sendBase("getPackets", args);
    }
    public List<ByteBuffer> recv_getPackets() throws org.apache.thrift.TException
    {
      getPackets_result result = new getPackets_result();
      receiveBase(result, "getPackets");
      if (result.isSetSuccess()) {
        return result.success;
      }
      throw new org.apache.thrift.TApplicationException(org.apache.thrift.TApplicationException.MISSING_RESULT, "getPackets failed: unknown result");
    }
    public int pushMessageSync(Message packet) throws org.apache.thrift.TException
    {
      send_pushMessageSync(packet);
      return recv_pushMessageSync();
    }
    public void send_pushMessageSync(Message packet) throws org.apache.thrift.TException
    {
      pushMessageSync_args args = new pushMessageSync_args();
      args.setPacket(packet);
      sendBase("pushMessageSync", args);
    }
    public int recv_pushMessageSync() throws org.apache.thrift.TException
    {
      pushMessageSync_result result = new pushMessageSync_result();
      receiveBase(result, "pushMessageSync");
      if (result.isSetSuccess()) {
        return result.success;
      }
      throw new org.apache.thrift.TApplicationException(org.apache.thrift.TApplicationException.MISSING_RESULT, "pushMessageSync failed: unknown result");
    }
    public void pushMessageAsync(Message packet) throws org.apache.thrift.TException
    {
      send_pushMessageAsync(packet);
    }
    public void send_pushMessageAsync(Message packet) throws org.apache.thrift.TException
    {
      pushMessageAsync_args args = new pushMessageAsync_args();
      args.setPacket(packet);
      sendBase("pushMessageAsync", args);
    }
    public void terminateSession(String sessionid) throws org.apache.thrift.TException
    {
      send_terminateSession(sessionid);
      recv_terminateSession();
    }
    public void send_terminateSession(String sessionid) throws org.apache.thrift.TException
    {
      terminateSession_args args = new terminateSession_args();
      args.setSessionid(sessionid);
      sendBase("terminateSession", args);
    }
    public void recv_terminateSession() throws org.apache.thrift.TException
    {
      terminateSession_result result = new terminateSession_result();
      receiveBase(result, "terminateSession");
      return;
    }
  }
  public static class AsyncClient extends org.apache.thrift.async.TAsyncClient implements AsyncIface {
    public static class Factory implements org.apache.thrift.async.TAsyncClientFactory<AsyncClient> {
      private org.apache.thrift.async.TAsyncClientManager clientManager;
      private org.apache.thrift.protocol.TProtocolFactory protocolFactory;
      public Factory(org.apache.thrift.async.TAsyncClientManager clientManager, org.apache.thrift.protocol.TProtocolFactory protocolFactory) {
        this.clientManager = clientManager;
        this.protocolFactory = protocolFactory;
      }
      public AsyncClient getAsyncClient(org.apache.thrift.transport.TNonblockingTransport transport) {
        return new AsyncClient(protocolFactory, clientManager, transport);
      }
    }
    public AsyncClient(org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.async.TAsyncClientManager clientManager, org.apache.thrift.transport.TNonblockingTransport transport) {
      super(protocolFactory, clientManager, transport);
    }
    public void getPackets(String sessionid, org.apache.thrift.async.AsyncMethodCallback<getPackets_call> resultHandler) throws org.apache.thrift.TException {
      checkReady();
      getPackets_call method_call = new getPackets_call(sessionid, resultHandler, this, ___protocolFactory, ___transport);
      this.___currentMethod = method_call;
      ___manager.call(method_call);
    }
    public static class getPackets_call extends org.apache.thrift.async.TAsyncMethodCall {
      private String sessionid;
      public getPackets_call(String sessionid, org.apache.thrift.async.AsyncMethodCallback<getPackets_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
        super(client, protocolFactory, transport, resultHandler, false);
        this.sessionid = sessionid;
      }
      public void write_args(org.apache.thrift.protocol.TProtocol prot) throws org.apache.thrift.TException {
        prot.writeMessageBegin(new org.apache.thrift.protocol.TMessage("getPackets", org.apache.thrift.protocol.TMessageType.CALL, 0));
        getPackets_args args = new getPackets_args();
        args.setSessionid(sessionid);
        args.write(prot);
        prot.writeMessageEnd();
      }
      public List<ByteBuffer> getResult() throws org.apache.thrift.TException {
        if (getState() != org.apache.thrift.async.TAsyncMethodCall.State.RESPONSE_READ) {
          throw new IllegalStateException("Method call not finished!");
        }
        org.apache.thrift.transport.TMemoryInputTransport memoryTransport = new org.apache.thrift.transport.TMemoryInputTransport(getFrameBuffer().array());
        org.apache.thrift.protocol.TProtocol prot = client.getProtocolFactory().getProtocol(memoryTransport);
        return (new Client(prot)).recv_getPackets();
      }
    }
    public void pushMessageSync(Message packet, org.apache.thrift.async.AsyncMethodCallback<pushMessageSync_call> resultHandler) throws org.apache.thrift.TException {
      checkReady();
      pushMessageSync_call method_call = new pushMessageSync_call(packet, resultHandler, this, ___protocolFactory, ___transport);
      this.___currentMethod = method_call;
      ___manager.call(method_call);
    }
    public static class pushMessageSync_call extends org.apache.thrift.async.TAsyncMethodCall {
      private Message packet;
      public pushMessageSync_call(Message packet, org.apache.thrift.async.AsyncMethodCallback<pushMessageSync_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
        super(client, protocolFactory, transport, resultHandler, false);
        this.packet = packet;
      }
      public void write_args(org.apache.thrift.protocol.TProtocol prot) throws org.apache.thrift.TException {
        prot.writeMessageBegin(new org.apache.thrift.protocol.TMessage("pushMessageSync", org.apache.thrift.protocol.TMessageType.CALL, 0));
        pushMessageSync_args args = new pushMessageSync_args();
        args.setPacket(packet);
        args.write(prot);
        prot.writeMessageEnd();
      }
      public int getResult() throws org.apache.thrift.TException {
        if (getState() != org.apache.thrift.async.TAsyncMethodCall.State.RESPONSE_READ) {
          throw new IllegalStateException("Method call not finished!");
        }
        org.apache.thrift.transport.TMemoryInputTransport memoryTransport = new org.apache.thrift.transport.TMemoryInputTransport(getFrameBuffer().array());
        org.apache.thrift.protocol.TProtocol prot = client.getProtocolFactory().getProtocol(memoryTransport);
        return (new Client(prot)).recv_pushMessageSync();
      }
    }
    public void pushMessageAsync(Message packet, org.apache.thrift.async.AsyncMethodCallback<pushMessageAsync_call> resultHandler) throws org.apache.thrift.TException {
      checkReady();
      pushMessageAsync_call method_call = new pushMessageAsync_call(packet, resultHandler, this, ___protocolFactory, ___transport);
      this.___currentMethod = method_call;
      ___manager.call(method_call);
    }
    public static class pushMessageAsync_call extends org.apache.thrift.async.TAsyncMethodCall {
      private Message packet;
      public pushMessageAsync_call(Message packet, org.apache.thrift.async.AsyncMethodCallback<pushMessageAsync_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
        super(client, protocolFactory, transport, resultHandler, true);
        this.packet = packet;
      }
      public void write_args(org.apache.thrift.protocol.TProtocol prot) throws org.apache.thrift.TException {
        prot.writeMessageBegin(new org.apache.thrift.protocol.TMessage("pushMessageAsync", org.apache.thrift.protocol.TMessageType.CALL, 0));
        pushMessageAsync_args args = new pushMessageAsync_args();
        args.setPacket(packet);
        args.write(prot);
        prot.writeMessageEnd();
      }
      public void getResult() throws org.apache.thrift.TException {
        if (getState() != org.apache.thrift.async.TAsyncMethodCall.State.RESPONSE_READ) {
          throw new IllegalStateException("Method call not finished!");
        }
        org.apache.thrift.transport.TMemoryInputTransport memoryTransport = new org.apache.thrift.transport.TMemoryInputTransport(getFrameBuffer().array());
        org.apache.thrift.protocol.TProtocol prot = client.getProtocolFactory().getProtocol(memoryTransport);
      }
    }
    public void terminateSession(String sessionid, org.apache.thrift.async.AsyncMethodCallback<terminateSession_call> resultHandler) throws org.apache.thrift.TException {
      checkReady();
      terminateSession_call method_call = new terminateSession_call(sessionid, resultHandler, this, ___protocolFactory, ___transport);
      this.___currentMethod = method_call;
      ___manager.call(method_call);
    }
    public static class terminateSession_call extends org.apache.thrift.async.TAsyncMethodCall {
      private String sessionid;
      public terminateSession_call(String sessionid, org.apache.thrift.async.AsyncMethodCallback<terminateSession_call> resultHandler, org.apache.thrift.async.TAsyncClient client, org.apache.thrift.protocol.TProtocolFactory protocolFactory, org.apache.thrift.transport.TNonblockingTransport transport) throws org.apache.thrift.TException {
        super(client, protocolFactory, transport, resultHandler, false);
        this.sessionid = sessionid;
      }
      public void write_args(org.apache.thrift.protocol.TProtocol prot) throws org.apache.thrift.TException {
        prot.writeMessageBegin(new org.apache.thrift.protocol.TMessage("terminateSession", org.apache.thrift.protocol.TMessageType.CALL, 0));
        terminateSession_args args = new terminateSession_args();
        args.setSessionid(sessionid);
        args.write(prot);
        prot.writeMessageEnd();
      }
      public void getResult() throws org.apache.thrift.TException {
        if (getState() != org.apache.thrift.async.TAsyncMethodCall.State.RESPONSE_READ) {
          throw new IllegalStateException("Method call not finished!");
        }
        org.apache.thrift.transport.TMemoryInputTransport memoryTransport = new org.apache.thrift.transport.TMemoryInputTransport(getFrameBuffer().array());
        org.apache.thrift.protocol.TProtocol prot = client.getProtocolFactory().getProtocol(memoryTransport);
        (new Client(prot)).recv_terminateSession();
      }
    }
  }
  public static class Processor<I extends Iface> extends org.apache.thrift.TBaseProcessor<I> implements org.apache.thrift.TProcessor {
    private static final Logger LOGGER = LoggerFactory.getLogger(Processor.class.getName());
    public Processor(I iface) {
      super(iface, getProcessMap(new HashMap<String, org.apache.thrift.ProcessFunction<I, ? extends org.apache.thrift.TBase>>()));
    }
    protected Processor(I iface, Map<String,  org.apache.thrift.ProcessFunction<I, ? extends  org.apache.thrift.TBase>> processMap) {
      super(iface, getProcessMap(processMap));
    }
    private static <I extends Iface> Map<String,  org.apache.thrift.ProcessFunction<I, ? extends  org.apache.thrift.TBase>> getProcessMap(Map<String,  org.apache.thrift.ProcessFunction<I, ? extends  org.apache.thrift.TBase>> processMap) {
      processMap.put("getPackets", new getPackets());
      processMap.put("pushMessageSync", new pushMessageSync());
      processMap.put("pushMessageAsync", new pushMessageAsync());
      processMap.put("terminateSession", new terminateSession());
      return processMap;
    }
    public static class getPackets<I extends Iface> extends org.apache.thrift.ProcessFunction<I, getPackets_args> {
      public getPackets() {
        super("getPackets");
      }
      public getPackets_args getEmptyArgsInstance() {
        return new getPackets_args();
      }
      protected boolean isOneway() {
        return false;
      }
      public getPackets_result getResult(I iface, getPackets_args args) throws org.apache.thrift.TException {
        getPackets_result result = new getPackets_result();
        result.success = iface.getPackets(args.sessionid);
        return result;
      }
    }
    public static class pushMessageSync<I extends Iface> extends org.apache.thrift.ProcessFunction<I, pushMessageSync_args> {
      public pushMessageSync() {
        super("pushMessageSync");
      }
      public pushMessageSync_args getEmptyArgsInstance() {
        return new pushMessageSync_args();
      }
      protected boolean isOneway() {
        return false;
      }
      public pushMessageSync_result getResult(I iface, pushMessageSync_args args) throws org.apache.thrift.TException {
        pushMessageSync_result result = new pushMessageSync_result();
        result.success = iface.pushMessageSync(args.packet);
        result.setSuccessIsSet(true);
        return result;
      }
    }
    public static class pushMessageAsync<I extends Iface> extends org.apache.thrift.ProcessFunction<I, pushMessageAsync_args> {
      public pushMessageAsync() {
        super("pushMessageAsync");
      }
      public pushMessageAsync_args getEmptyArgsInstance() {
        return new pushMessageAsync_args();
      }
      protected boolean isOneway() {
        return true;
      }
      public org.apache.thrift.TBase getResult(I iface, pushMessageAsync_args args) throws org.apache.thrift.TException {
        iface.pushMessageAsync(args.packet);
        return null;
      }
    }
    public static class terminateSession<I extends Iface> extends org.apache.thrift.ProcessFunction<I, terminateSession_args> {
      public terminateSession() {
        super("terminateSession");
      }
      public terminateSession_args getEmptyArgsInstance() {
        return new terminateSession_args();
      }
      protected boolean isOneway() {
        return false;
      }
      public terminateSession_result getResult(I iface, terminateSession_args args) throws org.apache.thrift.TException {
        terminateSession_result result = new terminateSession_result();
        iface.terminateSession(args.sessionid);
        return result;
      }
    }
  }
  public static class getPackets_args implements org.apache.thrift.TBase<getPackets_args, getPackets_args._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("getPackets_args");
    private static final org.apache.thrift.protocol.TField SESSIONID_FIELD_DESC = new org.apache.thrift.protocol.TField("sessionid", org.apache.thrift.protocol.TType.STRING, (short)1);
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new getPackets_argsStandardSchemeFactory());
      schemes.put(TupleScheme.class, new getPackets_argsTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
      SESSIONID((short)1, "sessionid");
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
            return SESSIONID;
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
      tmpMap.put(_Fields.SESSIONID, new org.apache.thrift.meta_data.FieldMetaData("sessionid", org.apache.thrift.TFieldRequirementType.DEFAULT, 
          new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING)));
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(getPackets_args.class, metaDataMap);
    }
    public getPackets_args() {
    }
    public getPackets_args(
      String sessionid)
    {
      this();
      this.sessionid = sessionid;
    }
    public getPackets_args(getPackets_args other) {
      if (other.isSetSessionid()) {
        this.sessionid = other.sessionid;
      }
    }
    public getPackets_args deepCopy() {
      return new getPackets_args(this);
    }
    @Override
    public void clear() {
      this.sessionid = null;
    }
    public String getSessionid() {
      return this.sessionid;
    }
    public getPackets_args setSessionid(String sessionid) {
      this.sessionid = sessionid;
      return this;
    }
    public void unsetSessionid() {
      this.sessionid = null;
    }
    public boolean isSetSessionid() {
      return this.sessionid != null;
    }
    public void setSessionidIsSet(boolean value) {
      if (!value) {
        this.sessionid = null;
      }
    }
    public void setFieldValue(_Fields field, Object value) {
      switch (field) {
      case SESSIONID:
        if (value == null) {
          unsetSessionid();
        } else {
          setSessionid((String)value);
        }
        break;
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
      case SESSIONID:
        return getSessionid();
      }
      throw new IllegalStateException();
    }
    public boolean isSet(_Fields field) {
      if (field == null) {
        throw new IllegalArgumentException();
      }
      switch (field) {
      case SESSIONID:
        return isSetSessionid();
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof getPackets_args)
        return this.equals((getPackets_args)that);
      return false;
    }
    public boolean equals(getPackets_args that) {
      if (that == null)
        return false;
      boolean this_present_sessionid = true && this.isSetSessionid();
      boolean that_present_sessionid = true && that.isSetSessionid();
      if (this_present_sessionid || that_present_sessionid) {
        if (!(this_present_sessionid && that_present_sessionid))
          return false;
        if (!this.sessionid.equals(that.sessionid))
          return false;
      }
      return true;
    }
    @Override
    public int hashCode() {
      return 0;
    }
    public int compareTo(getPackets_args other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      getPackets_args typedOther = (getPackets_args)other;
      lastComparison = Boolean.valueOf(isSetSessionid()).compareTo(typedOther.isSetSessionid());
      if (lastComparison != 0) {
        return lastComparison;
      }
      if (isSetSessionid()) {
        lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.sessionid, typedOther.sessionid);
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
      StringBuilder sb = new StringBuilder("getPackets_args(");
      boolean first = true;
      sb.append("sessionid:");
      if (this.sessionid == null) {
        sb.append("null");
      } else {
        sb.append(this.sessionid);
      }
      first = false;
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
        read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
      } catch (org.apache.thrift.TException te) {
        throw new java.io.IOException(te);
      }
    }
    private static class getPackets_argsStandardSchemeFactory implements SchemeFactory {
      public getPackets_argsStandardScheme getScheme() {
        return new getPackets_argsStandardScheme();
      }
    }
    private static class getPackets_argsStandardScheme extends StandardScheme<getPackets_args> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, getPackets_args struct) throws org.apache.thrift.TException {
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
                struct.sessionid = iprot.readString();
                struct.setSessionidIsSet(true);
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
      public void write(org.apache.thrift.protocol.TProtocol oprot, getPackets_args struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        if (struct.sessionid != null) {
          oprot.writeFieldBegin(SESSIONID_FIELD_DESC);
          oprot.writeString(struct.sessionid);
          oprot.writeFieldEnd();
        }
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class getPackets_argsTupleSchemeFactory implements SchemeFactory {
      public getPackets_argsTupleScheme getScheme() {
        return new getPackets_argsTupleScheme();
      }
    }
    private static class getPackets_argsTupleScheme extends TupleScheme<getPackets_args> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, getPackets_args struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
        BitSet optionals = new BitSet();
        if (struct.isSetSessionid()) {
          optionals.set(0);
        }
        oprot.writeBitSet(optionals, 1);
        if (struct.isSetSessionid()) {
          oprot.writeString(struct.sessionid);
        }
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, getPackets_args struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
        BitSet incoming = iprot.readBitSet(1);
        if (incoming.get(0)) {
          struct.sessionid = iprot.readString();
          struct.setSessionidIsSet(true);
        }
      }
    }
  }
  public static class getPackets_result implements org.apache.thrift.TBase<getPackets_result, getPackets_result._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("getPackets_result");
    private static final org.apache.thrift.protocol.TField SUCCESS_FIELD_DESC = new org.apache.thrift.protocol.TField("success", org.apache.thrift.protocol.TType.LIST, (short)0);
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new getPackets_resultStandardSchemeFactory());
      schemes.put(TupleScheme.class, new getPackets_resultTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
      SUCCESS((short)0, "success");
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
            return SUCCESS;
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
      tmpMap.put(_Fields.SUCCESS, new org.apache.thrift.meta_data.FieldMetaData("success", org.apache.thrift.TFieldRequirementType.DEFAULT, 
          new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
              new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING              , true))));
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(getPackets_result.class, metaDataMap);
    }
    public getPackets_result() {
    }
    public getPackets_result(
      List<ByteBuffer> success)
    {
      this();
      this.success = success;
    }
    public getPackets_result(getPackets_result other) {
      if (other.isSetSuccess()) {
        List<ByteBuffer> __this__success = new ArrayList<ByteBuffer>();
        for (ByteBuffer other_element : other.success) {
          ByteBuffer temp_binary_element = org.apache.thrift.TBaseHelper.copyBinary(other_element);
;
          __this__success.add(temp_binary_element);
        }
        this.success = __this__success;
      }
    }
    public getPackets_result deepCopy() {
      return new getPackets_result(this);
    }
    @Override
    public void clear() {
      this.success = null;
    }
    public int getSuccessSize() {
      return (this.success == null) ? 0 : this.success.size();
    }
    public java.util.Iterator<ByteBuffer> getSuccessIterator() {
      return (this.success == null) ? null : this.success.iterator();
    }
    public void addToSuccess(ByteBuffer elem) {
      if (this.success == null) {
        this.success = new ArrayList<ByteBuffer>();
      }
      this.success.add(elem);
    }
    public List<ByteBuffer> getSuccess() {
      return this.success;
    }
    public getPackets_result setSuccess(List<ByteBuffer> success) {
      this.success = success;
      return this;
    }
    public void unsetSuccess() {
      this.success = null;
    }
    public boolean isSetSuccess() {
      return this.success != null;
    }
    public void setSuccessIsSet(boolean value) {
      if (!value) {
        this.success = null;
      }
    }
    public void setFieldValue(_Fields field, Object value) {
      switch (field) {
      case SUCCESS:
        if (value == null) {
          unsetSuccess();
        } else {
          setSuccess((List<ByteBuffer>)value);
        }
        break;
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
      case SUCCESS:
        return getSuccess();
      }
      throw new IllegalStateException();
    }
    public boolean isSet(_Fields field) {
      if (field == null) {
        throw new IllegalArgumentException();
      }
      switch (field) {
      case SUCCESS:
        return isSetSuccess();
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof getPackets_result)
        return this.equals((getPackets_result)that);
      return false;
    }
    public boolean equals(getPackets_result that) {
      if (that == null)
        return false;
      boolean this_present_success = true && this.isSetSuccess();
      boolean that_present_success = true && that.isSetSuccess();
      if (this_present_success || that_present_success) {
        if (!(this_present_success && that_present_success))
          return false;
        if (!this.success.equals(that.success))
          return false;
      }
      return true;
    }
    @Override
    public int hashCode() {
      return 0;
    }
    public int compareTo(getPackets_result other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      getPackets_result typedOther = (getPackets_result)other;
      lastComparison = Boolean.valueOf(isSetSuccess()).compareTo(typedOther.isSetSuccess());
      if (lastComparison != 0) {
        return lastComparison;
      }
      if (isSetSuccess()) {
        lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.success, typedOther.success);
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
      StringBuilder sb = new StringBuilder("getPackets_result(");
      boolean first = true;
      sb.append("success:");
      if (this.success == null) {
        sb.append("null");
      } else {
        sb.append(this.success);
      }
      first = false;
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
        read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
      } catch (org.apache.thrift.TException te) {
        throw new java.io.IOException(te);
      }
    }
    private static class getPackets_resultStandardSchemeFactory implements SchemeFactory {
      public getPackets_resultStandardScheme getScheme() {
        return new getPackets_resultStandardScheme();
      }
    }
    private static class getPackets_resultStandardScheme extends StandardScheme<getPackets_result> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, getPackets_result struct) throws org.apache.thrift.TException {
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
                  org.apache.thrift.protocol.TList _list8 = iprot.readListBegin();
                  struct.success = new ArrayList<ByteBuffer>(_list8.size);
                  for (int _i9 = 0; _i9 < _list8.size; ++_i9)
                  {
                    _elem10 = iprot.readBinary();
                    struct.success.add(_elem10);
                  }
                  iprot.readListEnd();
                }
                struct.setSuccessIsSet(true);
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
      public void write(org.apache.thrift.protocol.TProtocol oprot, getPackets_result struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        if (struct.success != null) {
          oprot.writeFieldBegin(SUCCESS_FIELD_DESC);
          {
            oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRING, struct.success.size()));
            for (ByteBuffer _iter11 : struct.success)
            {
              oprot.writeBinary(_iter11);
            }
            oprot.writeListEnd();
          }
          oprot.writeFieldEnd();
        }
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class getPackets_resultTupleSchemeFactory implements SchemeFactory {
      public getPackets_resultTupleScheme getScheme() {
        return new getPackets_resultTupleScheme();
      }
    }
    private static class getPackets_resultTupleScheme extends TupleScheme<getPackets_result> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, getPackets_result struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
        BitSet optionals = new BitSet();
        if (struct.isSetSuccess()) {
          optionals.set(0);
        }
        oprot.writeBitSet(optionals, 1);
        if (struct.isSetSuccess()) {
          {
            oprot.writeI32(struct.success.size());
            for (ByteBuffer _iter12 : struct.success)
            {
              oprot.writeBinary(_iter12);
            }
          }
        }
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, getPackets_result struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
        BitSet incoming = iprot.readBitSet(1);
        if (incoming.get(0)) {
          {
            org.apache.thrift.protocol.TList _list13 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRING, iprot.readI32());
            struct.success = new ArrayList<ByteBuffer>(_list13.size);
            for (int _i14 = 0; _i14 < _list13.size; ++_i14)
            {
              _elem15 = iprot.readBinary();
              struct.success.add(_elem15);
            }
          }
          struct.setSuccessIsSet(true);
        }
      }
    }
  }
  public static class pushMessageSync_args implements org.apache.thrift.TBase<pushMessageSync_args, pushMessageSync_args._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("pushMessageSync_args");
    private static final org.apache.thrift.protocol.TField PACKET_FIELD_DESC = new org.apache.thrift.protocol.TField("packet", org.apache.thrift.protocol.TType.STRUCT, (short)1);
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new pushMessageSync_argsStandardSchemeFactory());
      schemes.put(TupleScheme.class, new pushMessageSync_argsTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
      PACKET((short)1, "packet");
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
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
      tmpMap.put(_Fields.PACKET, new org.apache.thrift.meta_data.FieldMetaData("packet", org.apache.thrift.TFieldRequirementType.DEFAULT, 
          new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, Message.class)));
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(pushMessageSync_args.class, metaDataMap);
    }
    public pushMessageSync_args() {
    }
    public pushMessageSync_args(
      Message packet)
    {
      this();
      this.packet = packet;
    }
    public pushMessageSync_args(pushMessageSync_args other) {
      if (other.isSetPacket()) {
        this.packet = new Message(other.packet);
      }
    }
    public pushMessageSync_args deepCopy() {
      return new pushMessageSync_args(this);
    }
    @Override
    public void clear() {
      this.packet = null;
    }
    public Message getPacket() {
      return this.packet;
    }
    public pushMessageSync_args setPacket(Message packet) {
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
      case PACKET:
        if (value == null) {
          unsetPacket();
        } else {
          setPacket((Message)value);
        }
        break;
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
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
      case PACKET:
        return isSetPacket();
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof pushMessageSync_args)
        return this.equals((pushMessageSync_args)that);
      return false;
    }
    public boolean equals(pushMessageSync_args that) {
      if (that == null)
        return false;
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
    public int compareTo(pushMessageSync_args other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      pushMessageSync_args typedOther = (pushMessageSync_args)other;
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
      StringBuilder sb = new StringBuilder("pushMessageSync_args(");
      boolean first = true;
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
    private static class pushMessageSync_argsStandardSchemeFactory implements SchemeFactory {
      public pushMessageSync_argsStandardScheme getScheme() {
        return new pushMessageSync_argsStandardScheme();
      }
    }
    private static class pushMessageSync_argsStandardScheme extends StandardScheme<pushMessageSync_args> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, pushMessageSync_args struct) throws org.apache.thrift.TException {
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
                struct.packet = new Message();
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
      public void write(org.apache.thrift.protocol.TProtocol oprot, pushMessageSync_args struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        if (struct.packet != null) {
          oprot.writeFieldBegin(PACKET_FIELD_DESC);
          struct.packet.write(oprot);
          oprot.writeFieldEnd();
        }
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class pushMessageSync_argsTupleSchemeFactory implements SchemeFactory {
      public pushMessageSync_argsTupleScheme getScheme() {
        return new pushMessageSync_argsTupleScheme();
      }
    }
    private static class pushMessageSync_argsTupleScheme extends TupleScheme<pushMessageSync_args> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, pushMessageSync_args struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
        BitSet optionals = new BitSet();
        if (struct.isSetPacket()) {
          optionals.set(0);
        }
        oprot.writeBitSet(optionals, 1);
        if (struct.isSetPacket()) {
          struct.packet.write(oprot);
        }
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, pushMessageSync_args struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
        BitSet incoming = iprot.readBitSet(1);
        if (incoming.get(0)) {
          struct.packet = new Message();
          struct.packet.read(iprot);
          struct.setPacketIsSet(true);
        }
      }
    }
  }
  public static class pushMessageSync_result implements org.apache.thrift.TBase<pushMessageSync_result, pushMessageSync_result._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("pushMessageSync_result");
    private static final org.apache.thrift.protocol.TField SUCCESS_FIELD_DESC = new org.apache.thrift.protocol.TField("success", org.apache.thrift.protocol.TType.I32, (short)0);
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new pushMessageSync_resultStandardSchemeFactory());
      schemes.put(TupleScheme.class, new pushMessageSync_resultTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
      SUCCESS((short)0, "success");
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
            return SUCCESS;
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
    private static final int __SUCCESS_ISSET_ID = 0;
    private byte __isset_bitfield = 0;
    public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
    static {
      Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
      tmpMap.put(_Fields.SUCCESS, new org.apache.thrift.meta_data.FieldMetaData("success", org.apache.thrift.TFieldRequirementType.DEFAULT, 
          new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32)));
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(pushMessageSync_result.class, metaDataMap);
    }
    public pushMessageSync_result() {
    }
    public pushMessageSync_result(
      int success)
    {
      this();
      this.success = success;
      setSuccessIsSet(true);
    }
    public pushMessageSync_result(pushMessageSync_result other) {
      __isset_bitfield = other.__isset_bitfield;
      this.success = other.success;
    }
    public pushMessageSync_result deepCopy() {
      return new pushMessageSync_result(this);
    }
    @Override
    public void clear() {
      setSuccessIsSet(false);
      this.success = 0;
    }
    public int getSuccess() {
      return this.success;
    }
    public pushMessageSync_result setSuccess(int success) {
      this.success = success;
      setSuccessIsSet(true);
      return this;
    }
    public void unsetSuccess() {
      __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __SUCCESS_ISSET_ID);
    }
    public boolean isSetSuccess() {
      return EncodingUtils.testBit(__isset_bitfield, __SUCCESS_ISSET_ID);
    }
    public void setSuccessIsSet(boolean value) {
      __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __SUCCESS_ISSET_ID, value);
    }
    public void setFieldValue(_Fields field, Object value) {
      switch (field) {
      case SUCCESS:
        if (value == null) {
          unsetSuccess();
        } else {
          setSuccess((Integer)value);
        }
        break;
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
      case SUCCESS:
        return Integer.valueOf(getSuccess());
      }
      throw new IllegalStateException();
    }
    public boolean isSet(_Fields field) {
      if (field == null) {
        throw new IllegalArgumentException();
      }
      switch (field) {
      case SUCCESS:
        return isSetSuccess();
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof pushMessageSync_result)
        return this.equals((pushMessageSync_result)that);
      return false;
    }
    public boolean equals(pushMessageSync_result that) {
      if (that == null)
        return false;
      boolean this_present_success = true;
      boolean that_present_success = true;
      if (this_present_success || that_present_success) {
        if (!(this_present_success && that_present_success))
          return false;
        if (this.success != that.success)
          return false;
      }
      return true;
    }
    @Override
    public int hashCode() {
      return 0;
    }
    public int compareTo(pushMessageSync_result other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      pushMessageSync_result typedOther = (pushMessageSync_result)other;
      lastComparison = Boolean.valueOf(isSetSuccess()).compareTo(typedOther.isSetSuccess());
      if (lastComparison != 0) {
        return lastComparison;
      }
      if (isSetSuccess()) {
        lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.success, typedOther.success);
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
      StringBuilder sb = new StringBuilder("pushMessageSync_result(");
      boolean first = true;
      sb.append("success:");
      sb.append(this.success);
      first = false;
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
    private static class pushMessageSync_resultStandardSchemeFactory implements SchemeFactory {
      public pushMessageSync_resultStandardScheme getScheme() {
        return new pushMessageSync_resultStandardScheme();
      }
    }
    private static class pushMessageSync_resultStandardScheme extends StandardScheme<pushMessageSync_result> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, pushMessageSync_result struct) throws org.apache.thrift.TException {
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
                struct.success = iprot.readI32();
                struct.setSuccessIsSet(true);
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
      public void write(org.apache.thrift.protocol.TProtocol oprot, pushMessageSync_result struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        if (struct.isSetSuccess()) {
          oprot.writeFieldBegin(SUCCESS_FIELD_DESC);
          oprot.writeI32(struct.success);
          oprot.writeFieldEnd();
        }
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class pushMessageSync_resultTupleSchemeFactory implements SchemeFactory {
      public pushMessageSync_resultTupleScheme getScheme() {
        return new pushMessageSync_resultTupleScheme();
      }
    }
    private static class pushMessageSync_resultTupleScheme extends TupleScheme<pushMessageSync_result> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, pushMessageSync_result struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
        BitSet optionals = new BitSet();
        if (struct.isSetSuccess()) {
          optionals.set(0);
        }
        oprot.writeBitSet(optionals, 1);
        if (struct.isSetSuccess()) {
          oprot.writeI32(struct.success);
        }
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, pushMessageSync_result struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
        BitSet incoming = iprot.readBitSet(1);
        if (incoming.get(0)) {
          struct.success = iprot.readI32();
          struct.setSuccessIsSet(true);
        }
      }
    }
  }
  public static class pushMessageAsync_args implements org.apache.thrift.TBase<pushMessageAsync_args, pushMessageAsync_args._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("pushMessageAsync_args");
    private static final org.apache.thrift.protocol.TField PACKET_FIELD_DESC = new org.apache.thrift.protocol.TField("packet", org.apache.thrift.protocol.TType.STRUCT, (short)1);
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new pushMessageAsync_argsStandardSchemeFactory());
      schemes.put(TupleScheme.class, new pushMessageAsync_argsTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
      PACKET((short)1, "packet");
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
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
      tmpMap.put(_Fields.PACKET, new org.apache.thrift.meta_data.FieldMetaData("packet", org.apache.thrift.TFieldRequirementType.DEFAULT, 
          new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, Message.class)));
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(pushMessageAsync_args.class, metaDataMap);
    }
    public pushMessageAsync_args() {
    }
    public pushMessageAsync_args(
      Message packet)
    {
      this();
      this.packet = packet;
    }
    public pushMessageAsync_args(pushMessageAsync_args other) {
      if (other.isSetPacket()) {
        this.packet = new Message(other.packet);
      }
    }
    public pushMessageAsync_args deepCopy() {
      return new pushMessageAsync_args(this);
    }
    @Override
    public void clear() {
      this.packet = null;
    }
    public Message getPacket() {
      return this.packet;
    }
    public pushMessageAsync_args setPacket(Message packet) {
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
      case PACKET:
        if (value == null) {
          unsetPacket();
        } else {
          setPacket((Message)value);
        }
        break;
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
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
      case PACKET:
        return isSetPacket();
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof pushMessageAsync_args)
        return this.equals((pushMessageAsync_args)that);
      return false;
    }
    public boolean equals(pushMessageAsync_args that) {
      if (that == null)
        return false;
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
    public int compareTo(pushMessageAsync_args other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      pushMessageAsync_args typedOther = (pushMessageAsync_args)other;
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
      StringBuilder sb = new StringBuilder("pushMessageAsync_args(");
      boolean first = true;
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
    private static class pushMessageAsync_argsStandardSchemeFactory implements SchemeFactory {
      public pushMessageAsync_argsStandardScheme getScheme() {
        return new pushMessageAsync_argsStandardScheme();
      }
    }
    private static class pushMessageAsync_argsStandardScheme extends StandardScheme<pushMessageAsync_args> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, pushMessageAsync_args struct) throws org.apache.thrift.TException {
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
                struct.packet = new Message();
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
      public void write(org.apache.thrift.protocol.TProtocol oprot, pushMessageAsync_args struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        if (struct.packet != null) {
          oprot.writeFieldBegin(PACKET_FIELD_DESC);
          struct.packet.write(oprot);
          oprot.writeFieldEnd();
        }
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class pushMessageAsync_argsTupleSchemeFactory implements SchemeFactory {
      public pushMessageAsync_argsTupleScheme getScheme() {
        return new pushMessageAsync_argsTupleScheme();
      }
    }
    private static class pushMessageAsync_argsTupleScheme extends TupleScheme<pushMessageAsync_args> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, pushMessageAsync_args struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
        BitSet optionals = new BitSet();
        if (struct.isSetPacket()) {
          optionals.set(0);
        }
        oprot.writeBitSet(optionals, 1);
        if (struct.isSetPacket()) {
          struct.packet.write(oprot);
        }
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, pushMessageAsync_args struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
        BitSet incoming = iprot.readBitSet(1);
        if (incoming.get(0)) {
          struct.packet = new Message();
          struct.packet.read(iprot);
          struct.setPacketIsSet(true);
        }
      }
    }
  }
  public static class terminateSession_args implements org.apache.thrift.TBase<terminateSession_args, terminateSession_args._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("terminateSession_args");
    private static final org.apache.thrift.protocol.TField SESSIONID_FIELD_DESC = new org.apache.thrift.protocol.TField("sessionid", org.apache.thrift.protocol.TType.STRING, (short)1);
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new terminateSession_argsStandardSchemeFactory());
      schemes.put(TupleScheme.class, new terminateSession_argsTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
      SESSIONID((short)1, "sessionid");
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
            return SESSIONID;
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
      tmpMap.put(_Fields.SESSIONID, new org.apache.thrift.meta_data.FieldMetaData("sessionid", org.apache.thrift.TFieldRequirementType.DEFAULT, 
          new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING)));
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(terminateSession_args.class, metaDataMap);
    }
    public terminateSession_args() {
    }
    public terminateSession_args(
      String sessionid)
    {
      this();
      this.sessionid = sessionid;
    }
    public terminateSession_args(terminateSession_args other) {
      if (other.isSetSessionid()) {
        this.sessionid = other.sessionid;
      }
    }
    public terminateSession_args deepCopy() {
      return new terminateSession_args(this);
    }
    @Override
    public void clear() {
      this.sessionid = null;
    }
    public String getSessionid() {
      return this.sessionid;
    }
    public terminateSession_args setSessionid(String sessionid) {
      this.sessionid = sessionid;
      return this;
    }
    public void unsetSessionid() {
      this.sessionid = null;
    }
    public boolean isSetSessionid() {
      return this.sessionid != null;
    }
    public void setSessionidIsSet(boolean value) {
      if (!value) {
        this.sessionid = null;
      }
    }
    public void setFieldValue(_Fields field, Object value) {
      switch (field) {
      case SESSIONID:
        if (value == null) {
          unsetSessionid();
        } else {
          setSessionid((String)value);
        }
        break;
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
      case SESSIONID:
        return getSessionid();
      }
      throw new IllegalStateException();
    }
    public boolean isSet(_Fields field) {
      if (field == null) {
        throw new IllegalArgumentException();
      }
      switch (field) {
      case SESSIONID:
        return isSetSessionid();
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof terminateSession_args)
        return this.equals((terminateSession_args)that);
      return false;
    }
    public boolean equals(terminateSession_args that) {
      if (that == null)
        return false;
      boolean this_present_sessionid = true && this.isSetSessionid();
      boolean that_present_sessionid = true && that.isSetSessionid();
      if (this_present_sessionid || that_present_sessionid) {
        if (!(this_present_sessionid && that_present_sessionid))
          return false;
        if (!this.sessionid.equals(that.sessionid))
          return false;
      }
      return true;
    }
    @Override
    public int hashCode() {
      return 0;
    }
    public int compareTo(terminateSession_args other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      terminateSession_args typedOther = (terminateSession_args)other;
      lastComparison = Boolean.valueOf(isSetSessionid()).compareTo(typedOther.isSetSessionid());
      if (lastComparison != 0) {
        return lastComparison;
      }
      if (isSetSessionid()) {
        lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.sessionid, typedOther.sessionid);
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
      StringBuilder sb = new StringBuilder("terminateSession_args(");
      boolean first = true;
      sb.append("sessionid:");
      if (this.sessionid == null) {
        sb.append("null");
      } else {
        sb.append(this.sessionid);
      }
      first = false;
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
        read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
      } catch (org.apache.thrift.TException te) {
        throw new java.io.IOException(te);
      }
    }
    private static class terminateSession_argsStandardSchemeFactory implements SchemeFactory {
      public terminateSession_argsStandardScheme getScheme() {
        return new terminateSession_argsStandardScheme();
      }
    }
    private static class terminateSession_argsStandardScheme extends StandardScheme<terminateSession_args> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, terminateSession_args struct) throws org.apache.thrift.TException {
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
                struct.sessionid = iprot.readString();
                struct.setSessionidIsSet(true);
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
      public void write(org.apache.thrift.protocol.TProtocol oprot, terminateSession_args struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        if (struct.sessionid != null) {
          oprot.writeFieldBegin(SESSIONID_FIELD_DESC);
          oprot.writeString(struct.sessionid);
          oprot.writeFieldEnd();
        }
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class terminateSession_argsTupleSchemeFactory implements SchemeFactory {
      public terminateSession_argsTupleScheme getScheme() {
        return new terminateSession_argsTupleScheme();
      }
    }
    private static class terminateSession_argsTupleScheme extends TupleScheme<terminateSession_args> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, terminateSession_args struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
        BitSet optionals = new BitSet();
        if (struct.isSetSessionid()) {
          optionals.set(0);
        }
        oprot.writeBitSet(optionals, 1);
        if (struct.isSetSessionid()) {
          oprot.writeString(struct.sessionid);
        }
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, terminateSession_args struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
        BitSet incoming = iprot.readBitSet(1);
        if (incoming.get(0)) {
          struct.sessionid = iprot.readString();
          struct.setSessionidIsSet(true);
        }
      }
    }
  }
  public static class terminateSession_result implements org.apache.thrift.TBase<terminateSession_result, terminateSession_result._Fields>, java.io.Serializable, Cloneable   {
    private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("terminateSession_result");
    private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
    static {
      schemes.put(StandardScheme.class, new terminateSession_resultStandardSchemeFactory());
      schemes.put(TupleScheme.class, new terminateSession_resultTupleSchemeFactory());
    }
    public enum _Fields implements org.apache.thrift.TFieldIdEnum {
;
      private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();
      static {
        for (_Fields field : EnumSet.allOf(_Fields.class)) {
          byName.put(field.getFieldName(), field);
        }
      }
      public static _Fields findByThriftId(int fieldId) {
        switch(fieldId) {
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
      metaDataMap = Collections.unmodifiableMap(tmpMap);
      org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(terminateSession_result.class, metaDataMap);
    }
    public terminateSession_result() {
    }
    public terminateSession_result(terminateSession_result other) {
    }
    public terminateSession_result deepCopy() {
      return new terminateSession_result(this);
    }
    @Override
    public void clear() {
    }
    public void setFieldValue(_Fields field, Object value) {
      switch (field) {
      }
    }
    public Object getFieldValue(_Fields field) {
      switch (field) {
      }
      throw new IllegalStateException();
    }
    public boolean isSet(_Fields field) {
      if (field == null) {
        throw new IllegalArgumentException();
      }
      switch (field) {
      }
      throw new IllegalStateException();
    }
    @Override
    public boolean equals(Object that) {
      if (that == null)
        return false;
      if (that instanceof terminateSession_result)
        return this.equals((terminateSession_result)that);
      return false;
    }
    public boolean equals(terminateSession_result that) {
      if (that == null)
        return false;
      return true;
    }
    @Override
    public int hashCode() {
      return 0;
    }
    public int compareTo(terminateSession_result other) {
      if (!getClass().equals(other.getClass())) {
        return getClass().getName().compareTo(other.getClass().getName());
      }
      int lastComparison = 0;
      terminateSession_result typedOther = (terminateSession_result)other;
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
      StringBuilder sb = new StringBuilder("terminateSession_result(");
      boolean first = true;
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
        read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
      } catch (org.apache.thrift.TException te) {
        throw new java.io.IOException(te);
      }
    }
    private static class terminateSession_resultStandardSchemeFactory implements SchemeFactory {
      public terminateSession_resultStandardScheme getScheme() {
        return new terminateSession_resultStandardScheme();
      }
    }
    private static class terminateSession_resultStandardScheme extends StandardScheme<terminateSession_result> {
      public void read(org.apache.thrift.protocol.TProtocol iprot, terminateSession_result struct) throws org.apache.thrift.TException {
        org.apache.thrift.protocol.TField schemeField;
        iprot.readStructBegin();
        while (true)
        {
          schemeField = iprot.readFieldBegin();
          if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
            break;
          }
          switch (schemeField.id) {
            default:
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
          }
          iprot.readFieldEnd();
        }
        iprot.readStructEnd();
        struct.validate();
      }
      public void write(org.apache.thrift.protocol.TProtocol oprot, terminateSession_result struct) throws org.apache.thrift.TException {
        struct.validate();
        oprot.writeStructBegin(STRUCT_DESC);
        oprot.writeFieldStop();
        oprot.writeStructEnd();
      }
    }
    private static class terminateSession_resultTupleSchemeFactory implements SchemeFactory {
      public terminateSession_resultTupleScheme getScheme() {
        return new terminateSession_resultTupleScheme();
      }
    }
    private static class terminateSession_resultTupleScheme extends TupleScheme<terminateSession_result> {
      @Override
      public void write(org.apache.thrift.protocol.TProtocol prot, terminateSession_result struct) throws org.apache.thrift.TException {
        TTupleProtocol oprot = (TTupleProtocol) prot;
      }
      @Override
      public void read(org.apache.thrift.protocol.TProtocol prot, terminateSession_result struct) throws org.apache.thrift.TException {
        TTupleProtocol iprot = (TTupleProtocol) prot;
      }
    }
  }
}
