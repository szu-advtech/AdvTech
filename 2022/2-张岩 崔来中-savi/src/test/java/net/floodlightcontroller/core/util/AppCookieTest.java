package net.floodlightcontroller.core.util;
import org.junit.Test;
import net.floodlightcontroller.core.util.AppCookie;
import net.floodlightcontroller.core.util.AppIDInUseException;
import net.floodlightcontroller.core.util.AppIDNotRegisteredException;
import net.floodlightcontroller.core.util.InvalidAppIDValueException;
import org.projectfloodlight.openflow.types.U64;
public class AppCookieTest {
    private static int appId = 0xF42;
    private static int appId2 = 0x743;
    private static int invalidAppId1 = 0x1000;
    private static int invalidAppId2 = -1;
    @Test
    public void testAppCookie(){
        String name = "FooBar";
        String name2 = "FooFooFoo";
        try {
            AppCookie.makeCookie(appId, user);
            fail("Expected exception not thrown");
        AppCookie.registerApp(appId, name);
        U64 cookie = AppCookie.makeCookie(appId, user);
        assertEquals(expectedCookie11, cookie);
        assertEquals(appId, AppCookie.extractApp(cookie));
        assertEquals(user, AppCookie.extractUser(cookie));
        cookie = AppCookie.makeCookie(appId, user2);
        assertEquals(expectedCookie12, cookie);
        assertEquals(appId, AppCookie.extractApp(cookie));
        assertEquals(user2, AppCookie.extractUser(cookie));
        AppCookie.registerApp(appId, name);
        try {
            AppCookie.registerApp(appId, name + "XXXXX");
            fail("Expected exception not thrown");
        try {
            AppCookie.makeCookie(appId2, user);
            fail("Expected exception not thrown");
        AppCookie.registerApp(appId2, name2);
        cookie = AppCookie.makeCookie(appId2, user);
        assertEquals(expectedCookie21, cookie);
        assertEquals(appId2, AppCookie.extractApp(cookie));
        assertEquals(user, AppCookie.extractUser(cookie));
        cookie = AppCookie.makeCookie(appId2, user2);
        assertEquals(expectedCookie22, cookie);
        assertEquals(appId2, AppCookie.extractApp(cookie));
        assertEquals(user2, AppCookie.extractUser(cookie));
        try {
            AppCookie.registerApp(invalidAppId1, "invalid");
            fail("Expected exception not thrown");
        try {
            AppCookie.registerApp(invalidAppId2, "also invalid");
            fail("Expected exception not thrown");
    }
}
