package org.sdnplatform.sync.internal;
import java.io.File;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Random;
import org.sdnplatform.sync.internal.util.ByteArray;
import org.sdnplatform.sync.internal.version.VectorClock;
public class TUtils {
    public static final String DIGITS = "0123456789";
    public static final String LETTERS = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM";
    public static final Random SEEDED_RANDOM = new Random(19873482374L);
    public static final Random UNSEEDED_RANDOM = new Random();
    public static VectorClock getClock(int... nodes) {
        VectorClock clock = new VectorClock();
        return increment(clock, nodes);
    }
    public static VectorClock getClockT(long timestamp, int... nodes) {
        VectorClock clock = new VectorClock(timestamp);
        return incrementT(timestamp, clock, nodes);
    }
    public static VectorClock incrementT(long timestamp, 
                                         VectorClock clock, int... nodes) {
        for(int n: nodes)
            clock = clock.incremented((short) n, timestamp);
        return clock;
    }
    public static VectorClock increment(VectorClock clock, int... nodes) {
        for(int n: nodes)
            clock = clock.incremented((short) n, System.currentTimeMillis());
        return clock;
    }
    public static boolean bytesEqual(byte[] a1, byte[] a2) {
        if(a1 == a2) {
            return true;
        } else if(a1 == null || a2 == null) {
            return false;
        } else if(a1.length != a2.length) {
            return false;
        } else {
            for(int i = 0; i < a1.length; i++)
                if(a1[i] != a2[i])
                    return false;
        }
        return true;
    }
    public static String randomLetters(int length) {
        return randomString(LETTERS, length);
    }
    public static String randomString(String sampler, int length) {
        StringBuilder builder = new StringBuilder(length);
        for(int i = 0; i < length; i++)
            builder.append(sampler.charAt(SEEDED_RANDOM.nextInt(sampler.length())));
        return builder.toString();
    }
    public static byte[] randomBytes(int length) {
        byte[] bytes = new byte[length];
        SEEDED_RANDOM.nextBytes(bytes);
        return bytes;
    }
    public static int[] randomInts(int max, int count) {
        int[] vals = new int[count];
        for(int i = 0; i < count; i++)
            vals[i] = SEEDED_RANDOM.nextInt(max);
        return vals;
    }
    public static int[] shuffle(int[] input) {
        List<Integer> vals = new ArrayList<Integer>(input.length);
        for(int i = 0; i < input.length; i++)
            vals.add(input[i]);
        Collections.shuffle(vals, SEEDED_RANDOM);
        int[] copy = new int[input.length];
        for(int i = 0; i < input.length; i++)
            copy[i] = vals.get(i);
        return copy;
    }
    public static long quantile(long[] values, double quantile) {
        if(values == null)
            throw new IllegalArgumentException("Values cannot be null.");
        if(quantile < 0.0 || quantile > 1.0)
            throw new IllegalArgumentException("Quantile must be between 0.0 and 1.0");
        long[] copy = new long[values.length];
        System.arraycopy(values, 0, copy, 0, copy.length);
        Arrays.sort(copy);
        return copy[index];
    }
    public static double mean(long[] values) {
        double total = 0.0;
        for(int i = 0; i < values.length; i++)
            total += values[i];
        return total / values.length;
    }
    public static File createTempDir() {
        return createTempDir(new File(System.getProperty("java.io.tmpdir")));
    }
    public static File createTempDir(File parent) {
        File temp = new File(parent,
                             Integer.toString(Math.abs(UNSEEDED_RANDOM.nextInt()) % 1000000));
        temp.delete();
        temp.mkdir();
        temp.deleteOnExit();
        return temp;
    }
    public static String quote(String s) {
        return "\"" + s + "\"";
    }
    public static ByteArray toByteArray(String s) {
        try {
            return new ByteArray(s.getBytes("UTF-8"));
        } catch(UnsupportedEncodingException e) {
            throw new IllegalStateException(e);
        }
    }
    public static void assertWithBackoff(long timeout, Attempt attempt) throws Exception {
        assertWithBackoff(30, timeout, attempt);
    }
    public static void assertWithBackoff(long initialDelay, long timeout, Attempt attempt)
            throws Exception {
        long delay = initialDelay;
        long finishBy = System.currentTimeMillis() + timeout;
        while(true) {
            try {
                attempt.checkCondition();
                return;
            } catch(AssertionError e) {
                if(System.currentTimeMillis() < finishBy) {
                    Thread.sleep(delay);
                } else {
                    throw e;
                }
            }
        }
    }
    @SuppressWarnings("unchecked")
    public static <T> T getPrivateValue(Object instance, String fieldName) throws Exception {
        Field eventDataQueueField = instance.getClass().getDeclaredField(fieldName);
        eventDataQueueField.setAccessible(true);
        return (T) eventDataQueueField.get(instance);
    }
    public static GregorianCalendar getCalendar(int year,
                                                int month,
                                                int day,
                                                int hour,
                                                int mins,
                                                int secs) {
        GregorianCalendar cal = new GregorianCalendar();
        cal.set(Calendar.YEAR, year);
        cal.set(Calendar.MONTH, month);
        cal.set(Calendar.DATE, day);
        cal.set(Calendar.HOUR_OF_DAY, hour);
        cal.set(Calendar.MINUTE, mins);
        cal.set(Calendar.SECOND, secs);
        cal.set(Calendar.MILLISECOND, 0);
        return cal;
    }
}
