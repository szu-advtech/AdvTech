
import java.util.ArrayList;
import java.util.HashMap;

public class Initialization
{
    public static void initialization()
	{

        Data.U = new float[Data.n+1][Data.d];
        
    	// ======================================================    	

    	for (int u=1; u<Data.n+1; u++)
    	{
    		for (int f=0; f<Data.d; f++)
    		{
    			Data.U[u][f] = (float) ( (Math.random()-0.5)*0.01 );
    		}
    	}

    	// ====================================================== 
    	Data.userSimilarityMatrix = new float[Data.n + 1][Data.n + 1];
    	calculateSimilarity();
    	System.out.println("Finished calculating the similarity");
    }
    
	private static void calculateSimilarity() {
		


		for (int u1=1; u1<Data.n+1; u1++) {
			for (int u2=1; u2<Data.n+1; u2++) {
				if(u1 != u2 && Data.userSimilarityMatrix[u1][u2] == 0.0) 
				{
					float coRatedItemsNum = 0;
					for (Integer i1 : Data.I_u[u1]) {
						if(Data.I_u[u2].contains(i1)) coRatedItemsNum++;
					}
					
					if(coRatedItemsNum > Data.threshold)
					{
						float similarity = (float) (coRatedItemsNum / Math.sqrt(Data.I_u[u1].size() * Data.I_u[u2].size()));
						Data.userSimilarityMatrix[u1][u2] = similarity;
						Data.userSimilarityMatrix[u2][u1] = similarity;

					
						if(Data.neighborhoodHashMap.containsKey(u1))
						{
							ArrayList<Integer> userList = Data.neighborhoodHashMap.get(u1);
							userList.add(u2);
							Data.neighborhoodHashMap.put(u1, userList);
						}
						else
						{
							ArrayList<Integer> userList = new ArrayList<Integer>();
							userList.add(u2);
							Data.neighborhoodHashMap.put(u1, userList);
						}
						
						
						if(Data.neighborhoodHashMap.containsKey(u2))
						{
							ArrayList<Integer> userList = Data.neighborhoodHashMap.get(u2);
							userList.add(u1);
							Data.neighborhoodHashMap.put(u2, userList);
						}
						else
						{
							ArrayList<Integer> userList = new ArrayList<Integer>();
							userList.add(u1);
							Data.neighborhoodHashMap.put(u2, userList);
						}
			
					}
				}
			}
		}


		
	}


	private static HashMap<Integer, ArrayList<Integer>> put(
			ArrayList<Integer> userList) {
		// TODO Auto-generated method stub
		return null;
	}
}
 