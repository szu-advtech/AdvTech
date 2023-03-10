package org.sdnplatform.sync.thrift;
import java.util.Map;
import java.util.HashMap;
import org.apache.thrift.TEnum;
@SuppressWarnings("all") public enum MessageType implements org.apache.thrift.TEnum {
  HELLO(1),
  ERROR(2),
  ECHO_REQUEST(3),
  ECHO_REPLY(4),
  GET_REQUEST(5),
  GET_RESPONSE(6),
  PUT_REQUEST(7),
  PUT_RESPONSE(8),
  DELETE_REQUEST(9),
  DELETE_RESPONSE(10),
  SYNC_VALUE(11),
  SYNC_VALUE_RESPONSE(12),
  SYNC_OFFER(13),
  SYNC_REQUEST(14),
  FULL_SYNC_REQUEST(15),
  CURSOR_REQUEST(16),
  CURSOR_RESPONSE(17),
  REGISTER_REQUEST(18),
  REGISTER_RESPONSE(19),
  CLUSTER_JOIN_REQUEST(20),
  CLUSTER_JOIN_RESPONSE(21);
  private final int value;
  private MessageType(int value) {
    this.value = value;
  }
  public int getValue() {
    return value;
  }
  public static MessageType findByValue(int value) { 
    switch (value) {
      case 1:
        return HELLO;
      case 2:
        return ERROR;
      case 3:
        return ECHO_REQUEST;
      case 4:
        return ECHO_REPLY;
      case 5:
        return GET_REQUEST;
      case 6:
        return GET_RESPONSE;
      case 7:
        return PUT_REQUEST;
      case 8:
        return PUT_RESPONSE;
      case 9:
        return DELETE_REQUEST;
      case 10:
        return DELETE_RESPONSE;
      case 11:
        return SYNC_VALUE;
      case 12:
        return SYNC_VALUE_RESPONSE;
      case 13:
        return SYNC_OFFER;
      case 14:
        return SYNC_REQUEST;
      case 15:
        return FULL_SYNC_REQUEST;
      case 16:
        return CURSOR_REQUEST;
      case 17:
        return CURSOR_RESPONSE;
      case 18:
        return REGISTER_REQUEST;
      case 19:
        return REGISTER_RESPONSE;
      case 20:
        return CLUSTER_JOIN_REQUEST;
      case 21:
        return CLUSTER_JOIN_RESPONSE;
      default:
        return null;
    }
  }
}
