package net.floodlightcontroller.core.web;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
public class HealthCheckResource extends ServerResource {
    public static class HealthCheckInfo {
        protected boolean healthy;
        public HealthCheckInfo() {
            this.healthy = true;
        }
        public boolean isHealthy() {
            return healthy;
        }
        public void setHealthy(boolean healthy) {
            this.healthy = healthy;
        }
    }
    @Get("json")
    public HealthCheckInfo healthCheck() {
        return new HealthCheckInfo();
    }
}
