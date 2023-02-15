
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

public class Data 
{
	// === Configurations	
	public static int d = 20; 
	
	public static float lambda = 0.01f;
	

	public static float gamma = 0.5f;
	

	public static String fnTrainingData = "";
	public static String fnTestData = "";
	public static String fnOutputData = "";
    
	public static int n = 943; 
	public static int m = 1682; 
	public static int num_test; 
	public static float MinRating = 1.0f; 
	public static float MaxRating = 5.0f; 
	

	public static int num_iterations = 100;
    

	public static int rating_num = 0;
	

	public static Set<Integer> trainUserNo;

	public static HashMap<Integer, HashMap<Integer, Double>> traningDataMap = new HashMap<Integer, HashMap<Integer, Double>>();
     

	public static int[] indexUserTest;
	public static int[] indexItemTest;
	public static float[] ratingTest;


	public static float[][] userSimilarityMatrix;


	public static float threshold = 0;


	public static HashSet<Integer>[] I_u; 

	public static float[][] U;
	

	public static Client client[];
	
	public static CountDownLatch synchronize;

	public static FileWriter fw ;
	public static BufferedWriter bw;
	
	
	public static HashMap<Integer, Integer> neighborHashMap = new HashMap<Integer, Integer>();
	
	public static HashMap<Integer, ArrayList<Integer>> neighborhoodHashMap = new HashMap<Integer, ArrayList<Integer>>();

} 