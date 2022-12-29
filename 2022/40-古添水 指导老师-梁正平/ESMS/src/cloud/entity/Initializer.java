import java.util.ArrayList;
import java.util.List;

public class Initializer {
    public static List<VM> initVMSet() {
        List<VM> vms = new ArrayList<>();
        int type = 0;
        while(type < VM.TYPE_NO) {
            VM vm = new VM(type);
            vms.add(vm);
            type++;
        }
        return vms;
    }
}
