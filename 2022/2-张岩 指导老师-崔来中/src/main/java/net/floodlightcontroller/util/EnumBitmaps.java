package net.floodlightcontroller.util;
import java.util.EnumSet;
import java.util.Set;
public class EnumBitmaps {
    public interface BitmapableEnum {
        int getValue();
    }
    public static <E extends Enum<E> & BitmapableEnum>
            EnumSet<E> toEnumSet(Class<E> type, int bitmap) {
        if (type == null)
            throw new NullPointerException("Given enum type must not be null");
        EnumSet<E> s = EnumSet.noneOf(type);
        int allSetBitmap = 0;
        for (E element: type.getEnumConstants()) {
            if (Integer.bitCount(element.getValue()) != 1) {
                String msg = String.format("The %s (%x) constant of the " +
                        "enum %s is supposed to represent a bitmap entry but " +
                        "has more than one bit set.",
                        element.toString(), element.getValue(), type.getName());
                throw new IllegalArgumentException(msg);
            }
            allSetBitmap |= element.getValue();
            if ((bitmap & element.getValue()) != 0)
                s.add(element);
        }
        if (((~allSetBitmap) & bitmap) != 0) {
            String msg = String.format("The bitmap %x for enum %s has " +
                    "bits set that are presented by any enum constant",
                    bitmap, type.getName());
            throw new IllegalArgumentException(msg);
        }
        return s;
    }
    public static <E extends Enum<E> & BitmapableEnum>
            int getMask(Class<E> type) {
        if (type == null)
            throw new NullPointerException("Given enum type must not be null");
        int allSetBitmap = 0;
        for (E element: type.getEnumConstants()) {
            if (Integer.bitCount(element.getValue()) != 1) {
                String msg = String.format("The %s (%x) constant of the " +
                        "enum %s is supposed to represent a bitmap entry but " +
                        "has more than one bit set.",
                        element.toString(), element.getValue(), type.getName());
                throw new IllegalArgumentException(msg);
            }
            allSetBitmap |= element.getValue();
        }
        return allSetBitmap;
    }
    public static <E extends Enum<E> & BitmapableEnum>
            int toBitmap(Set<E> set) {
        if (set == null)
            throw new NullPointerException("Given set must not be null");
        int bitmap = 0;
        for (E element: set) {
            if (Integer.bitCount(element.getValue()) != 1) {
                String msg = String.format("The %s (%x) constant in the set " +
                        "is supposed to represent a bitmap entry but " +
                        "has more than one bit set.",
                        element.toString(), element.getValue());
                throw new IllegalArgumentException(msg);
            }
            bitmap |= element.getValue();
        }
        return bitmap;
    }
}
