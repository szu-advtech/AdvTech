
import java.io.*;
import java.util.Arrays;

public class ReadConfigurations
{
	public static void readConfigurations(String[]args) throws IOException 
	{
        // ==========================================================
		// read the configurations
		for (int k=0; k < args.length; k++)
		{
			if (args[k].equals("-d")) Data.d = Integer.parseInt(args[++k]);
			else if (args[k].equals("-lambda")) Data.lambda = Float.parseFloat(args[++k]);        		
			else if (args[k].equals("-gamma")) Data.gamma = Float.parseFloat(args[++k]);
			else if (args[k].equals("-fnTrainingData")) Data.fnTrainingData = args[++k];
			else if (args[k].equals("-fnTestData")) Data.fnTestData = args[++k];
			else if (args[k].equals("-fnOutputData")) Data.fnOutputData = args[++k];
			else if (args[k].equals("-MinRating")) Data.MinRating = Float.parseFloat(args[++k]);
			else if (args[k].equals("-MaxRating")) Data.MaxRating = Float.parseFloat(args[++k]);    		
			else if (args[k].equals("-n")) Data.n = Integer.parseInt(args[++k]);
			else if (args[k].equals("-m")) Data.m = Integer.parseInt(args[++k]);
			else if (args[k].equals("-num_iterations")) Data.num_iterations = Integer.parseInt(args[++k]);

		}
        // ==========================================================
        
        // ==========================================================
		// print the configurations
		System.out.println(Arrays.toString(args));
		System.out.println("d: " + Integer.toString(Data.d));    	
		System.out.println("lambda: " + Float.toString(Data.lambda));
		System.out.println("gamma: " + Float.toString(Data.gamma));    		
		System.out.println("fnTrainingData: " + Data.fnTrainingData);
		System.out.println("fnTestData: " + Data.fnTestData);
		System.out.println("fnOutputData: " + Data.fnOutputData);    	
		System.out.println("MinRating: " + Float.toString(Data.MinRating));
		System.out.println("MaxRating: " + Float.toString(Data.MaxRating));
		System.out.println("n: " + Integer.toString(Data.n));
		System.out.println("m: " + Integer.toString(Data.m));    	    	
		System.out.println("num_iterations: " + Integer.toString(Data.num_iterations));
        // ==========================================================
        
        // ==========================================================
		try {
			// --- initialization of file operation
			File file = new File(Data.fnOutputData);
			if(!file.exists())
			{
				file.createNewFile();
			}
			Data.fw = new FileWriter(Data.fnOutputData);
			Data.bw = new BufferedWriter(Data.fw);
			
			// --- output the value of parameters to file
			Data.bw.write("d: " + Integer.toString(Data.d) + "\r\n");
			Data.bw.write("lambda: " + Float.toString(Data.lambda) + "\r\n");
			Data.bw.write("gamma: " + Float.toString(Data.gamma) + "\r\n");
			Data.bw.write("fnTrainingData: " + Data.fnTrainingData + "\r\n");
			Data.bw.write("fnTestData: " + Data.fnTestData + "\r\n");
			Data.bw.write("fnOutputData: " + Data.fnOutputData + "\r\n");
			Data.bw.write("MinRating: " + Float.toString(Data.MinRating) + "\r\n");
			Data.bw.write("MaxRating: " + Float.toString(Data.MaxRating) + "\r\n");
			Data.bw.write("n: " + Integer.toString(Data.n) + "\r\n");
			Data.bw.write("m: " + Integer.toString(Data.m) + "\r\n");
			Data.bw.write("num_iterations: " + Integer.toString(Data.num_iterations) + "\r\n");
			Data.bw.flush();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        // ==========================================================
	}
}
 