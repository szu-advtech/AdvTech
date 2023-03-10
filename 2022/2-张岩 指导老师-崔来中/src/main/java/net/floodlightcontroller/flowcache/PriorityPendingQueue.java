package net.floodlightcontroller.flowcache;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
@Deprecated
public class PriorityPendingQueue<E> {
    private LinkedBlockingQueue<E> highPriorityQueue;
    private LinkedBlockingQueue<E> mediumPriorityQueue;
    private LinkedBlockingQueue<E> lowPriorityQueue;
    private final AtomicInteger count = new AtomicInteger(0);
    private final ReentrantLock takeLock = new ReentrantLock();
    private final Condition notEmpty = takeLock.newCondition();
    private final ReentrantLock putLock = new ReentrantLock();
    private final Condition notFull = putLock.newCondition();
    private final int capacity;
    public enum EventPriority {
        HIGH,
        MEDIUM,
        LOW,
    }
    public PriorityPendingQueue() {
        highPriorityQueue=   new LinkedBlockingQueue<E>();
        mediumPriorityQueue= new LinkedBlockingQueue<E>();
        lowPriorityQueue=    new LinkedBlockingQueue<E>();
        capacity= Integer.MAX_VALUE;
    }
    public E take() throws InterruptedException {
        E x;
        int c = -1;
        final AtomicInteger count = this.count;
        final ReentrantLock takeLock = this.takeLock;
        takeLock.lockInterruptibly();
        try {
            try {
                while (count.get() == 0)
                    notEmpty.await();
            } catch (InterruptedException ie) {
                throw ie;
            }
            x = extract();
            c = count.getAndDecrement();
            if (c > 1)
               notEmpty.signal();
            } finally {
            takeLock.unlock();
        }
        if (c == capacity)
            signalNotFull();
        return x;
    }
    public E poll() {
        final AtomicInteger count = this.count;
        if (count.get() == 0)
            return null;
        E x = null;
        int c = -1;
        final ReentrantLock takeLock = this.takeLock;
        takeLock.lock();
        try {
            if (count.get() > 0) {
                x = extract();
                c = count.getAndDecrement();
                if (c > 1)
                    notEmpty.signal();
            }
        } finally {
            takeLock.unlock();
        }
        if (c == capacity)
            signalNotFull();
        return x;
    }
    public E peek() {
        return null;
        }
    public boolean offer(E e, EventPriority p) {
        if (e == null) throw new NullPointerException();
        final AtomicInteger count = this.count;
        if (count.get() == capacity)
            return false;
        int c = -1;
        final ReentrantLock putLock = this.putLock;
        putLock.lock();
        try {
            if (count.get() < capacity) {
                insert(e,p);
                c = count.getAndIncrement();
                if (c + 1 < capacity)
                    notFull.signal();
            }
        } finally {
            putLock.unlock();
        }
        if (c == 0)
            signalNotEmpty();
        return c >= 0;
    }
    public boolean offer(E e) {
        return false;
    }
    private E extract() {
        E first = highPriorityQueue.poll();
        if (first==null)
            first = mediumPriorityQueue.poll();
        if (first==null)
            first = lowPriorityQueue.poll();
        return first;
    }
    private boolean insert(E e, EventPriority p) {
        boolean result = false;
        switch (p) {
            case HIGH:
                result = highPriorityQueue.offer(e);
                break;
            case MEDIUM:
                result = mediumPriorityQueue.offer(e);
                break;
            case LOW:
                result = lowPriorityQueue.offer(e);
                break;
        }
        return result;
    }
    private void signalNotFull() {
         final ReentrantLock putLock = this.putLock;
         putLock.lock();
         try {
             notFull.signal();
         } finally {
             putLock.unlock();
         }
     }
    private void signalNotEmpty() {
         final ReentrantLock takeLock = this.takeLock;
         takeLock.lock();
         try {
             notEmpty.signal();
         } finally {
             takeLock.unlock();
         }
     }
    private void fullyLock() {
        putLock.lock();
        takeLock.lock();
    }
    private void fullyUnlock() {
        takeLock.unlock();
        putLock.unlock();
    }
    public int size() {
        return count.get();
    }
    public void clear() {
        fullyLock();
        try {
            highPriorityQueue.clear();
            mediumPriorityQueue.clear();
            lowPriorityQueue.clear();
            count.set(0);
        } finally {
            fullyUnlock();
        }
    }
    public boolean isEmpty() {
        return count.get() == 0;
    }
}