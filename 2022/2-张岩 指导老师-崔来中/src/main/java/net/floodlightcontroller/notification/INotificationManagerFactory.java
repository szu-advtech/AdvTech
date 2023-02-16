package net.floodlightcontroller.notification;
public interface INotificationManagerFactory {
    <T> INotificationManager getNotificationManager(Class<T> clazz);
}
