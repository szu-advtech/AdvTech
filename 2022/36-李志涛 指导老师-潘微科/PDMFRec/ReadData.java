
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

public class ReadData
{
    public static void readData() throws IOException 
	{

        Data.I_u = new HashSet[Data.n + 1];
        for (int u = 1; u <= Data.n; ++u) {
        	Data.I_u[u] = new HashSet<Integer>();
        }
        // ----------------------------------------------------
        
        // ==========================================================
        Data.trainUserNo = new HashSet<Integer>();
        // ----------------------------------------------------


        // ==========================================================
    	Data.num_test = 0;
    	BufferedReader brTest = new BufferedReader(new FileReader(Data.fnTestData));
    	String line = null;
    	while ((line = brTest.readLine())!=null)
    	{
    		Data.num_test += 1;
    	}
    	System.out.println("num_test: " + Data.num_test);
        // ----------------------------------------------------
    	

    	// ==========================================================


        // --- test data
    	Data.indexUserTest = new int[Data.num_test];
    	Data.indexItemTest = new int[Data.num_test];
    	Data.ratingTest = new float[Data.num_test];
        // ----------------------------------------------------

        

    	BufferedReader brTrain = new BufferedReader(new FileReader(Data.fnTrainingData));    	
    	line = null;
    	while ((line = brTrain.readLine())!=null)
    	{	
    		String[] terms = line.split("\\s+|,|;");
    		int userID = Integer.parseInt(terms[0]);
    		int itemID = Integer.parseInt(terms[1]);
    		float rating = Float.parseFloat(terms[2]);
//    		Data.r[userID][itemID] = rating;
    		Data.I_u[userID].add(itemID);  		
    		Data.rating_num++;
    		
    		if(Data.traningDataMap.containsKey(userID))
    		{
    			HashMap<Integer, Double> itemRatingMap = Data.traningDataMap.get(userID);
    			itemRatingMap.put(itemID, (double) rating);
    			Data.traningDataMap.put(userID, itemRatingMap);
    		}
    		else
    		{
    			HashMap<Integer, Double> itemRatingMap = new HashMap<Integer, Double>();
    			itemRatingMap.put(itemID, (double) rating);
    			Data.traningDataMap.put(userID, itemRatingMap);
    		}
    		
    		// ---
    		Data.trainUserNo.add(userID);
    
    	}
    	brTrain.close();
    	System.out.println("rating_num: " + Data.rating_num);
    	System.out.println("Finished reading the training data");
    	// ========================================================== 
    	
    	
        // ==========================================================  	
    	int id_case = 0; // initialize it to zero
    	brTest = new BufferedReader(new FileReader(Data.fnTestData));
    	line = null;
    	while ((line = brTest.readLine())!=null)
    	{	
    		String[] terms = line.split("\\s+|,|;");
    		int userID = Integer.parseInt(terms[0]);    		
    		int itemID = Integer.parseInt(terms[1]);
    		float rating = Float.parseFloat(terms[2]);
    		Data.indexUserTest[id_case] = userID;
    		Data.indexItemTest[id_case] = itemID;
    		Data.ratingTest[id_case] = rating;
    		id_case+=1;
    	}
    	brTest.close();
    	System.out.println("Finished reading the test data");
        // ----------------------------------------------------

    }    
}
 