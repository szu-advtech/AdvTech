package net.floodlightcontroller.core.util;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class SingletonTask {
    protected static final Logger logger = LoggerFactory.getLogger(SingletonTask.class);
    protected static class SingletonTaskContext  {
        protected boolean taskShouldRun = false;
        protected boolean taskRunning = false;
        protected SingletonTaskWorker waitingTask = null;
    }
    protected static class SingletonTaskWorker implements Runnable  {
        SingletonTask parent;
        boolean canceled = false;
        long nextschedule = 0;
        public SingletonTaskWorker(SingletonTask parent) {
            super();
            this.parent = parent;
        }
        @Override
        public void run() {
            synchronized (parent.context) {
                if (canceled || !parent.context.taskShouldRun) {
                    return;
                }
                parent.context.taskRunning = true;
                parent.context.taskShouldRun = false;
            }
            try {
                parent.task.run();
            } catch (Exception e) {
                logger.error("Exception while executing task", e);
            }
            catch (Error e) {
                logger.error("Error while executing task", e);
                throw e;
            }
            synchronized (parent.context) {
                parent.context.taskRunning = false;
                if (parent.context.taskShouldRun) {
                    long now = System.nanoTime();
                    if ((nextschedule <= 0 || (nextschedule - now) <= 0)) {
                        parent.ses.execute(this);
                    } else {
                        parent.ses.schedule(this, 
                                            nextschedule-now, 
                                            TimeUnit.NANOSECONDS);
                    }
                }
            }
        }
    }
    protected SingletonTaskContext context = new SingletonTaskContext();
    protected Runnable task;
    protected ScheduledExecutorService ses;
    public SingletonTask(ScheduledExecutorService ses,
            Runnable task) {
        super();
        this.task = task;
        this.ses = ses;
    }
    public void reschedule(long delay, TimeUnit unit) {
        boolean needQueue = true;
        SingletonTaskWorker stw = null;
        synchronized (context) {
            if (context.taskRunning || context.taskShouldRun) {
                if (context.taskRunning) {
                    if (delay > 0) {
                        long now = System.nanoTime();
                        long then = now + TimeUnit.NANOSECONDS.convert(delay, unit);
                        context.waitingTask.nextschedule = then;
                        logger.debug("rescheduled task " + this + " for " + TimeUnit.SECONDS.convert(then, TimeUnit.NANOSECONDS) + "s. A bunch of these messages -may- indicate you have a blocked task.");
                    } else {
                        context.waitingTask.nextschedule = 0;
                    }
                    needQueue = false;
                } else {
                    context.waitingTask.canceled = true;
                    context.waitingTask = null;
                }
            }
            context.taskShouldRun = true;
            if (needQueue) {
                stw = context.waitingTask = new SingletonTaskWorker(this);                    
            }
        }
        if (needQueue) {
            if (delay <= 0) {
                ses.execute(stw);
            } else {
                ses.schedule(stw, delay, unit);
            }
        }
    }
}