package net.floodlightcontroller.core;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.internal.CmdLineSettings;
import net.floodlightcontroller.core.module.FloodlightModuleConfigFileNotFoundException;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.FloodlightModuleLoader;
import net.floodlightcontroller.core.module.IFloodlightModuleContext;
import net.floodlightcontroller.restserver.IRestApiService;
public class Main {
	private static final Logger logger = LoggerFactory.getLogger(Main.class);
	public static void main(String[] args) throws FloodlightModuleException {
		try {
			System.setProperty("org.restlet.engine.loggerFacadeClass", 
					"org.restlet.ext.slf4j.Slf4jLoggerFacade");
			CmdLineSettings settings = new CmdLineSettings();
			CmdLineParser parser = new CmdLineParser(settings);
			try {
				parser.parseArgument(args);
			} catch (CmdLineException e) {
				parser.printUsage(System.out);
				System.exit(1);
			}
			FloodlightModuleLoader fml = new FloodlightModuleLoader();
			try {
				IFloodlightModuleContext moduleContext = fml.loadModulesFromConfig(settings.getModuleFile());
				IRestApiService restApi = moduleContext.getServiceImpl(IRestApiService.class);
				restApi.run(); 
			} catch (FloodlightModuleConfigFileNotFoundException e) {
				logger.error("Could not read config file: {}", e.getMessage());
				System.exit(1);
			}
			try {
            } catch (FloodlightModuleException e) {
                logger.error("Failed to run controller modules", e);
                System.exit(1);
            }
		} catch (Exception e) {
			logger.error("Exception in main", e);
			System.exit(1);
		}
	}
}
