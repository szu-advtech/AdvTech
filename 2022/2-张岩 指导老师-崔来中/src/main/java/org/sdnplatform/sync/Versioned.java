package org.sdnplatform.sync;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import org.sdnplatform.sync.IVersion.Occurred;
import org.sdnplatform.sync.internal.version.VectorClock;
import com.google.common.base.Objects;
public class Versioned<T> implements Serializable {
    private static final long serialVersionUID = 1;
    private volatile VectorClock version;
    private volatile T value;
    public Versioned(T object) {
        this(object, new VectorClock());
    }
    public Versioned(T object, 
                     IVersion version) {
        this.version = version == null ? new VectorClock() : (VectorClock) version;
        this.value = object;
    }
    public IVersion getVersion() {
        return version;
    }
    public void increment(int nodeId, long time) {
        this.version = version.incremented(nodeId, time);
    }
    public T getValue() {
        return value;
    }
    public void setValue(T object) {
        this.value = object;
    }
    private static boolean deepEquals(Object o1, Object o2) {
        if(o1 == o2) {
            return true;
        }
        if(o1 == null || o2 == null) {
            return false;
        }
        Class<?> type1 = o1.getClass();
        Class<?> type2 = o2.getClass();
        if(!(type1.isArray() && type2.isArray())) {
            return o1.equals(o2);
        }
        if(o1 instanceof Object[] && o2 instanceof Object[]) {
            return Arrays.deepEquals((Object[]) o1, (Object[]) o2);
        }
        if(type1 != type2) {
            return false;
        }
        if(o1 instanceof boolean[]) {
            return Arrays.equals((boolean[]) o1, (boolean[]) o2);
        }
        if(o1 instanceof char[]) {
            return Arrays.equals((char[]) o1, (char[]) o2);
        }
        if(o1 instanceof byte[]) {
            return Arrays.equals((byte[]) o1, (byte[]) o2);
        }
        if(o1 instanceof short[]) {
            return Arrays.equals((short[]) o1, (short[]) o2);
        }
        if(o1 instanceof int[]) {
            return Arrays.equals((int[]) o1, (int[]) o2);
        }
        if(o1 instanceof long[]) {
            return Arrays.equals((long[]) o1, (long[]) o2);
        }
        if(o1 instanceof float[]) {
            return Arrays.equals((float[]) o1, (float[]) o2);
        }
        if(o1 instanceof double[]) {
            return Arrays.equals((double[]) o1, (double[]) o2);
        }
        throw new AssertionError();
    }
    @Override
    public boolean equals(Object o) {
        if(o == this)
            return true;
        else if(!(o instanceof Versioned<?>))
            return false;
        Versioned<?> versioned = (Versioned<?>) o;
        return Objects.equal(getVersion(), versioned.getVersion())
               && deepEquals(getValue(), versioned.getValue());
    }
    @Override
    public int hashCode() {
        int v = 31 + version.hashCode();
        if(value != null) {
        }
        return v;
    }
    @Override
    public String toString() {
        return "[" + value + ", " + version + "]";
    }
    public Versioned<T> cloneVersioned() {
        return new Versioned<T>(this.getValue(), this.version.clone());
    }
    public static <S> Versioned<S> value(S s) {
        return new Versioned<S>(s, new VectorClock());
    }
    public static <S> Versioned<S> value(S s, IVersion v) {
        return new Versioned<S>(s, v);
    }
    public static <S> Versioned<S> emptyVersioned() {
        return new Versioned<S>(null, new VectorClock(0));
    }
    public static final class HappenedBeforeComparator<S> implements Comparator<Versioned<S>> {
        public int compare(Versioned<S> v1, Versioned<S> v2) {
            Occurred occurred = v1.getVersion().compare(v2.getVersion());
            if(occurred == Occurred.BEFORE)
                return -1;
            else if(occurred == Occurred.AFTER)
                return 1;
            else
                return 0;
        }
    }
}
