package net.floodlightcontroller.savi.analysis.web;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.savi.analysis.IAnalysisService;
public class AttackLogResource extends ServerResource {
	Logger log=LoggerFactory.getLogger(AttackLogResource.class);
	@Get("json")
	public Object retrieve() {
		StringBuffer attacklog = new StringBuffer();
		IAnalysisService analysisService=(IAnalysisService) getContext().getAttributes().get(IAnalysisService.class.getCanonicalName());
		 try {
             String encoding="utf-8";
             File file=new File(analysisService.getFilePath());
             if(file.isFile() && file.exists()){
                 InputStreamReader read = new InputStreamReader(
                 new FileInputStream(file),encoding);
                 BufferedReader bufferedReader = new BufferedReader(read);
                 String lineTxt = null;
                 while((lineTxt = bufferedReader.readLine()) != null){
                	 attacklog.append(lineTxt+'\n');
                 }
                 read.close();
             }else
 				log.info("no file");
	     } catch (Exception e) {
	    	 log.info("wrong");
	         e.printStackTrace();
	     }
		Map<String , String> logger= new HashMap<>(); 
		logger.put("logger", attacklog.toString());
		return logger;
	}
}
