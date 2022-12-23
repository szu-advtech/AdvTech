package org.sdnplatform.sync.thrift;
import java.util.Map;
import java.util.HashMap;
import org.apache.thrift.TEnum;
@SuppressWarnings("all") public enum AuthScheme implements org.apache.thrift.TEnum {
  NO_AUTH(0),
  CHALLENGE_RESPONSE(1);
  private final int value;
  private AuthScheme(int value) {
    this.value = value;
  }
  public int getValue() {
    return value;
  }
  public static AuthScheme findByValue(int value) { 
    switch (value) {
      case 0:
        return NO_AUTH;
      case 1:
        return CHALLENGE_RESPONSE;
      default:
        return null;
    }
  }
}
