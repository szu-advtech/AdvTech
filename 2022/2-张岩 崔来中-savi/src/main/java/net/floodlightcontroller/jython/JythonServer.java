package net.floodlightcontroller.jython;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import org.python.util.PythonInterpreter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class JythonServer extends Thread {
    protected static Logger log = LoggerFactory.getLogger(JythonServer.class);
    String host;
	int port;
	Map<String, Object> locals;
	public JythonServer(String host_, int port_, Map<String, Object> locals_) {
		this.host = host_;
		this.port = port_;
		this.locals = locals_;
		if (this.locals == null) {
			this.locals = new HashMap<String, Object>();
		}
		this.locals.put("log", JythonServer.log);
		this.setName("debugserver");
	}
    public void run() {
        PythonInterpreter p = new PythonInterpreter();
        for (String name : this.locals.keySet()) {
            p.set(name, this.locals.get(name));
        }
        URL jarUrl = JythonServer.class.getProtectionDomain().getCodeSource().getLocation();
        String jarPath = jarUrl.getPath();
        if (jarUrl.getProtocol().equals("file")) {
            jarPath = jarPath + "../../src/main/python/";
        }
        p.exec("import sys");
        p.exec("sys.path.append('" + jarPath + "')");
        p.exec("from debugserver import run_server");
        if (this.host == null) {
        	p.exec("run_server(port=" + this.port + ", locals=locals())");
        } else {
        	p.exec("run_server(port=" + this.port + ", host='" + this.host + "', locals=locals())");
        }
    }
}
