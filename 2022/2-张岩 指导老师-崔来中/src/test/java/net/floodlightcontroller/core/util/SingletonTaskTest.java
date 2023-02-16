package net.floodlightcontroller.core.util;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.junit.Before;
import org.junit.Test;
import net.floodlightcontroller.test.FloodlightTestCase;
public class SingletonTaskTest extends FloodlightTestCase {
    public int ran = 0;
    public int finished = 0;
    public long time = 0;
    @Override
    @Before
    public void setUp() throws Exception {
        super.setUp();
        ran = 0;
        finished = 0;
        time = 0;
    }
    @Test
    public void testBasic() throws InterruptedException {
        ScheduledExecutorService ses =
            Executors.newSingleThreadScheduledExecutor();
        SingletonTask st1 = new SingletonTask(ses, new Runnable() {
            @Override
            public void run() {
                ran += 1;
            }
        });
        st1.reschedule(0, null);
        ses.shutdown();
        ses.awaitTermination(5, TimeUnit.SECONDS);
        assertEquals("Check that task ran", 1, ran);
    }
    @Test
    public void testDelay() throws InterruptedException {
        ScheduledExecutorService ses =
            Executors.newSingleThreadScheduledExecutor();
        SingletonTask st1 = new SingletonTask(ses, new Runnable() {
            @Override
            public void run() {
                ran += 1;
                time = System.nanoTime();
            }
        });
        st1.reschedule(10, TimeUnit.MILLISECONDS);
        assertFalse("Check that task hasn't run yet", ran > 0);
        ses.shutdown();
        ses.awaitTermination(5, TimeUnit.SECONDS);
        assertEquals("Check that task ran", 1, ran);
    }
    @Test
    public void testReschedule() throws InterruptedException {
        ScheduledExecutorService ses =
            Executors.newSingleThreadScheduledExecutor();
        final Object tc = this;
        SingletonTask st1 = new SingletonTask(ses, new Runnable() {
            @Override
            public void run() {
                synchronized (tc) {
                    ran += 1;
                }
                time = System.nanoTime();
            }
        });
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        st1.reschedule(20, TimeUnit.MILLISECONDS);
        Thread.sleep(5);
        assertFalse("Check that task hasn't run yet", ran > 0);
        ses.shutdown();
        ses.awaitTermination(5, TimeUnit.SECONDS);
        assertEquals("Check that task ran only once", 1, ran);
    }
    @Test
    public void testConcurrentAddDelay() throws InterruptedException {
        ScheduledExecutorService ses =
            Executors.newSingleThreadScheduledExecutor();
        final Object tc = this;
        SingletonTask st1 = new SingletonTask(ses, new Runnable() {
            @Override
            public void run() {
                synchronized (tc) {
                    ran += 1;
                }
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (tc) {
                    finished += 1;
                    time = System.nanoTime();
                }
            }
        });
        st1.reschedule(5, TimeUnit.MILLISECONDS);
        Thread.sleep(20);
        assertEquals("Check that task started", 1, ran);
        assertEquals("Check that task not finished", 0, finished);
        st1.reschedule(75, TimeUnit.MILLISECONDS);
        assertTrue("Check task running state true", st1.context.taskRunning);
        assertTrue("Check task should run state true", st1.context.taskShouldRun);
        assertEquals("Check that task started", 1, ran);
        assertEquals("Check that task not finished", 0, finished);
        Thread.sleep(150);
        assertTrue("Check task running state false", !st1.context.taskRunning);
        assertTrue("Check task should run state false", !st1.context.taskShouldRun);
        assertEquals("Check that task ran exactly twice", 2, ran);
        assertEquals("Check that task finished exactly twice", 2, finished);
        ses.shutdown();
        ses.awaitTermination(15, TimeUnit.SECONDS);
    }
    @Test
    public void testConcurrentAddDelay2() throws InterruptedException {
        ScheduledExecutorService ses =
            Executors.newSingleThreadScheduledExecutor();
        final Object tc = this;
        SingletonTask st1 = new SingletonTask(ses, new Runnable() {
            @Override
            public void run() {
                synchronized (tc) {
                    ran += 1;
                }
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (tc) {
                    finished += 1;
                    time = System.nanoTime();
                }
            }
        });
        st1.reschedule(5, TimeUnit.MILLISECONDS);
        Thread.sleep(20);
        assertEquals("Check that task started", 1, ran);
        assertEquals("Check that task not finished", 0, finished);
        st1.reschedule(25, TimeUnit.MILLISECONDS);
        assertTrue("Check task running state true", st1.context.taskRunning);
        assertTrue("Check task should run state true", st1.context.taskShouldRun);
        assertEquals("Check that task started", 1, ran);
        assertEquals("Check that task not finished", 0, finished);
        Thread.sleep(150);
        assertTrue("Check task running state false", !st1.context.taskRunning);
        assertTrue("Check task should run state false", !st1.context.taskShouldRun);
        assertEquals("Check that task ran exactly twice", 2, ran);
        assertEquals("Check that task finished exactly twice", 2, finished);
        ses.shutdown();
        ses.awaitTermination(5, TimeUnit.SECONDS);
    }
    @Test
    public void testConcurrentAddNoDelay() throws InterruptedException {
        ScheduledExecutorService ses =
            Executors.newSingleThreadScheduledExecutor();
        final Object tc = this;
        SingletonTask st1 = new SingletonTask(ses, new Runnable() {
            @Override
            public void run() {
                synchronized (tc) {
                    ran += 1;
                }
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                synchronized (tc) {
                    finished += 1;
                    time = System.nanoTime();
                }
            }
        });
        st1.reschedule(0, null);
        Thread.sleep(20);
        assertEquals("Check that task started", 1, ran);
        assertEquals("Check that task not finished", 0, finished);
        st1.reschedule(0, null);
        assertTrue("Check task running state true", st1.context.taskRunning);
        assertTrue("Check task should run state true", st1.context.taskShouldRun);
        assertEquals("Check that task started", 1, ran);
        assertEquals("Check that task not finished", 0, finished);
        Thread.sleep(150);
        assertTrue("Check task running state false", !st1.context.taskRunning);
        assertTrue("Check task should run state false", !st1.context.taskShouldRun);
        assertEquals("Check that task ran exactly twice", 2, ran);
        assertEquals("Check that task finished exactly twice", 2, finished);
        ses.shutdown();
        ses.awaitTermination(5, TimeUnit.SECONDS);
    }
}
