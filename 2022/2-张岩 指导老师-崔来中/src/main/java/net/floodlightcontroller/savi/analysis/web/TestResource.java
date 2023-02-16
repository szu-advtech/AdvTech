package net.floodlightcontroller.savi.analysis.web;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
public class TestResource extends ServerResource {
	@Get("json")
	public Object test(){
		TestResult testResult=new TestResult("hello,world");
		return testResult;
	}
}
class TestResult{
	private String x;
	public String getX() {
		return x;
	}
	public void setX(String x) {
		this.x = x;
	}
	public TestResult(String x){
		this.x=x;
	}
}
