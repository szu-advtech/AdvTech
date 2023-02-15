import java.io.IOException;
import java.util.concurrent.*;

public class Main {

	public static void main(String[] args)  {
		
		try {
			// 1. read configurations
			ReadConfigurations.readConfigurations(args);
			
			// 2. read training data and test data 
	        ReadData.readData();
	               
			// 3. apply initialization
			Initialization.initialization();
	
			// 4. train
			Train t = new Train();
			t.train(Data.num_iterations);

			// 6. test
			Test.test();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (BrokenBarrierException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
} 
