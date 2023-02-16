package net.floodlightcontroller.core.web;
import net.floodlightcontroller.core.module.ModuleLoaderResource;
import net.floodlightcontroller.restserver.RestletRoutable;
import org.restlet.Context;
import org.restlet.Restlet;
import org.restlet.routing.Router;
public class CoreWebRoutable implements RestletRoutable {
	public static final String STR_SWITCH_ID = "switchId";
	public static final String STR_STAT_TYPE = "statType";
	public static final String STR_CTR_TITLE = "counterTitle";
	public static final String STR_CTR_MODULE = "counterModule";
	public static final String STR_LAYER = "layer";
	public static final String STR_ALL = "all";
	public static final String STR_ROLE = "role";
    @Override
    public String basePath() {
        return "/wm/core";
    }
    @Override
    public Restlet getRestlet(Context context) {
        Router router = new Router(context);
        router.attach("/module/all/json", ModuleLoaderResource.class);
        router.attach("/module/loaded/json", LoadedModuleLoaderResource.class);
        router.attach("/switch/{" + STR_SWITCH_ID + "}/role/json", SwitchRoleResource.class);
        router.attach("/switch/all/{" + STR_STAT_TYPE + "}/json", AllSwitchStatisticsResource.class);
        router.attach("/switch/{" + STR_SWITCH_ID + "}/{" + STR_STAT_TYPE + "}/json", SwitchStatisticsResource.class);
        router.attach("/controller/switches/json", ControllerSwitchesResource.class);
        router.attach("/counter/{" + STR_CTR_MODULE + "}/{" + STR_CTR_TITLE + "}/json", CounterResource.class);
        router.attach("/memory/json", ControllerMemoryResource.class);
        router.attach("/packettrace/json", PacketTraceResource.class);
        router.attach("/storage/tables/json", StorageSourceTablesResource.class);
        router.attach("/controller/summary/json", ControllerSummaryResource.class);
        router.attach("/role/json", ControllerRoleResource.class);
        router.attach("/health/json", HealthCheckResource.class);
        router.attach("/system/uptime/json", SystemUptimeResource.class);
        return router;
    }
}
