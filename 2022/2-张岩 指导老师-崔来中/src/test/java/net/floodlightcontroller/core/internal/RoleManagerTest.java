package net.floodlightcontroller.core.internal;
import static org.easymock.EasyMock.anyObject;
import org.junit.Before;
import org.junit.Test;
import net.floodlightcontroller.test.FloodlightTestCase;
import static org.easymock.EasyMock.createMock;
import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.expectLastCall;
import static org.easymock.EasyMock.replay;
import static org.easymock.EasyMock.reset;
import static org.easymock.EasyMock.verify;
import org.junit.After;
import net.floodlightcontroller.core.HARole;
import net.floodlightcontroller.core.IShutdownService;
import net.floodlightcontroller.core.internal.Controller.IUpdate;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.MockDebugCounterService;
public class RoleManagerTest extends FloodlightTestCase {
    private Controller controller;
    private RoleManager roleManager;
    @Override
    @Before
    public void setUp() throws Exception {
        doSetUp(HARole.ACTIVE);
    }
    private void doSetUp(HARole role) {
        controller = createMock(Controller.class);
        reset(controller);
        IDebugCounterService counterService = new MockDebugCounterService();
        expect(controller.getDebugCounter()).andReturn(counterService).anyTimes();
        replay(controller);
        IShutdownService shutdownService = createMock(IShutdownService.class);
        roleManager = new RoleManager(controller, shutdownService , role, "test");
        assertTrue(roleManager.getRole().equals(role));
    }
    @After
    public void tearDown() {
        verify(controller);
    }
    @Test
    public void testSetRoleStandbyToActive() throws Exception {
        doSetUp(HARole.STANDBY);
        this.setRoleAndMockController(HARole.ACTIVE);
        assertTrue(roleManager.getRole() == HARole.ACTIVE);
    }
    @Test
    public void testSetRoleActiveToStandby() throws Exception {
        assertTrue(roleManager.getRole() == HARole.ACTIVE);
        this.setRoleAndMockController(HARole.STANDBY);
        assertTrue(roleManager.getRole() == HARole.STANDBY);
    }
    @Test
    public void testSetRoleActiveToActive() throws Exception {
        assertTrue(roleManager.getRole() == HARole.ACTIVE);
        this.setRoleAndMockController(HARole.ACTIVE);
        assertTrue(roleManager.getRole() == HARole.ACTIVE);
    }
    @Test
    public void testSetRoleStandbyToStandby() throws Exception {
        doSetUp(HARole.STANDBY);
        this.setRoleAndMockController(HARole.STANDBY);
        assertTrue(roleManager.getRole() == HARole.STANDBY);
    }
    private void setRoleAndMockController(HARole role) {
        reset(controller);
        controller.addUpdateToQueue(anyObject(IUpdate.class));
        expectLastCall().anyTimes();
        replay(controller);
        roleManager.setRole(role, "test");
    }
}