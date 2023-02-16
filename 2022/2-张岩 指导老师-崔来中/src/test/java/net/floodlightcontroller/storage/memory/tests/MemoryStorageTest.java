package net.floodlightcontroller.storage.memory.tests;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.restserver.RestApiServer;
import net.floodlightcontroller.storage.memory.MemoryStorageSource;
import net.floodlightcontroller.storage.tests.StorageTest;
import org.junit.Before;
public class MemoryStorageTest extends StorageTest {
    @Before
    public void setUp() throws Exception {
        storageSource = new MemoryStorageSource();
        restApi = new RestApiServer();
        FloodlightModuleContext fmc = new FloodlightModuleContext();
        fmc.addService(IRestApiService.class, restApi);
        fmc.addService(IDebugCounterService.class, new MockDebugCounterService());
        restApi.init(fmc);
        storageSource.init(fmc);
        restApi.startUp(fmc);
        storageSource.startUp(fmc);
        super.setUp();
    }
}
