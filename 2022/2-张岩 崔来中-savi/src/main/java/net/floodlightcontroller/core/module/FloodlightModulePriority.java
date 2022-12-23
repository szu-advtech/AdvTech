package net.floodlightcontroller.core.module;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface FloodlightModulePriority {
    public enum Priority {
        MINIMUM(0),
        TEST(10),
        EXTRA_LOW(20),
        LOW(30),
        NORMAL(40),
        DEFAULT_PROVIDER(50),
        HIGH(60),
        EXTRA_HIGH(70);
        private final int value;
        private Priority(int value) {
            this.value = value;
        }
        public int value() {
            return value;
        }
    }
    public Priority value() default Priority.NORMAL;
}
