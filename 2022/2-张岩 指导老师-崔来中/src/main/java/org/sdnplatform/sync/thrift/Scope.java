package org.sdnplatform.sync.thrift;
import java.util.Map;
import java.util.HashMap;
import org.apache.thrift.TEnum;
@SuppressWarnings("all") public enum Scope implements org.apache.thrift.TEnum {
  GLOBAL(0),
  LOCAL(1),
  UNSYNCHRONIZED(2);
  private final int value;
  private Scope(int value) {
    this.value = value;
  }
  public int getValue() {
    return value;
  }
  public static Scope findByValue(int value) { 
    switch (value) {
      case 0:
        return GLOBAL;
      case 1:
        return LOCAL;
      case 2:
        return UNSYNCHRONIZED;
      default:
        return null;
    }
  }
}
