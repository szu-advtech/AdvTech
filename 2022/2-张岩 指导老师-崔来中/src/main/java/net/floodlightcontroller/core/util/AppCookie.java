package net.floodlightcontroller.core.util;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.projectfloodlight.openflow.types.U64;
public class AppCookie {
    static final int APP_ID_BITS = 12;
    static final long APP_ID_MASK = (1L << APP_ID_BITS) - 1;
    static final int APP_ID_SHIFT = (64 - APP_ID_BITS);
    static final long USER_MASK = 0x00000000FFFFFFFFL;
    static final int SRC_MAC_REWRITE_BIT=33;
    static final int DEST_MAC_REWRITE_BIT=34;
    static final int SRC_IP_REWRITE_BIT=35;
    static final int DEST_IP_REWRITE_BIT=36;
    static final long REWRITE_MASK= 0x000f00000000L;
    private static ConcurrentMap<Integer, String> appIdMap =
            new ConcurrentHashMap<Integer, String>();
    static public U64 makeCookie(int application, int user) {
        if (!appIdMap.containsKey(application)) {
            throw new AppIDNotRegisteredException(application);
        }
        long longApp = application;
        return U64.of((longApp << APP_ID_SHIFT) | longUser);
    }
    static public int extractApp(U64 cookie) {
        return (int)((cookie.getValue() >>> APP_ID_SHIFT) & APP_ID_MASK);
    }
    static public int extractUser(U64 cookie) {
        return (int)(cookie.getValue() & USER_MASK);
    }
    static public boolean isRewriteFlagSet(U64 cookie) {
        if ((cookie.getValue() & REWRITE_MASK) !=0L)
            return true;
        return false;
    }
    static public boolean isSrcMacRewriteFlagSet(U64 cookie) {
        if ((cookie.getValue() & (1L << (SRC_MAC_REWRITE_BIT-1))) !=0L)
            return true;
        return false;
    }
    static public boolean isDestMacRewriteFlagSet(U64 cookie) {
        if ((cookie.getValue() & (1L << (DEST_MAC_REWRITE_BIT-1))) !=0L)
            return true;
        return false;
    }
    static public boolean isSrcIpRewriteFlagSet(U64 cookie) {
        if ((cookie.getValue() & (1L << (SRC_IP_REWRITE_BIT-1))) !=0L)
            return true;
        return false;
    }
    static public boolean isDestIpRewriteFlagSet(U64 cookie) {
        if ((cookie.getValue() & (1L << (DEST_IP_REWRITE_BIT-1))) !=0L)
            return true;
        return false;
    }
    static public U64 setSrcMacRewriteFlag(U64 cookie) {
        return U64.of(cookie.getValue() | (1L << (SRC_MAC_REWRITE_BIT-1)));
    }
    static public U64 setDestMacRewriteFlag(U64 cookie) {
        return U64.of(cookie.getValue() | (1L << (DEST_MAC_REWRITE_BIT-1)));
    }
    static public U64 setSrcIpRewriteFlag(U64 cookie) {
        return U64.of(cookie.getValue() | (1L << (SRC_IP_REWRITE_BIT-1)));
    }
    static public U64 setDestIpRewriteFlag(U64 cookie) {
        return U64.of(cookie.getValue() | (1L << (DEST_IP_REWRITE_BIT-1)));
    }
    public static void registerApp(int application, String appName)
        throws AppIDException
    {
        if ((application & APP_ID_MASK) != application) {
            throw new InvalidAppIDValueException(application);
        }
        String oldApp = appIdMap.putIfAbsent(application, appName);
        if (oldApp != null && !oldApp.equals(appName)) {
            throw new AppIDInUseException(application, oldApp, appName);
        }
    }
    public static String getAppName(int application) {
        return appIdMap.get(application);
    }
}
