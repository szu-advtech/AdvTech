package net.floodlightcontroller.notification;
import net.floodlightcontroller.notification.syslog.SyslogNotificationFactory;
public class NotificationManagerFactory {
    public static final String  NOTIFICATION_FACTORY_NAME =
            "floodlight.notification.factoryName";
    private static INotificationManagerFactory factory; 
    static {
        NotificationManagerFactory.init();
    }
    protected static void init() {
        String notificationfactoryClassName = null;
        try {
            notificationfactoryClassName =
                    System.getProperty(NOTIFICATION_FACTORY_NAME);
        } catch (SecurityException e) {
            throw new RuntimeException(e);
        }
        if (notificationfactoryClassName != null) {
            Class<?> nfc;
            try {
                nfc = Class.forName(notificationfactoryClassName);
                factory = (INotificationManagerFactory) nfc.newInstance();
            } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
         } else {
         }
    }
    public static <T> INotificationManager getNotificationManager(Class<T> clazz) {
        return factory.getNotificationManager(clazz);
    }
    public static <T> INotificationManagerFactory getNotificationManagerFactory() {
        return factory;
    }
}
