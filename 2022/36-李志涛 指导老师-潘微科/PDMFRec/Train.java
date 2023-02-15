
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class Train{

	Train() {

	}

	public void train(int num_iterations) throws InterruptedException, BrokenBarrierException {
		// ==========================================================
		// --- Construct Clients
		Data.client = new Client[Data.n + 1];
		for (int u=1; u<Data.n+1; u++) {
			Data.client[u] = new Client(u);
		}

		// ----------------------------------------------------
		// --- Train
		for (int iter = 0; iter < num_iterations; iter++){

			// output each iteration result
			try {
				Data.bw.write("Iter:" + Integer.toString(iter) + "| ");
				Data.bw.flush();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			System.out.print("Iter:" + Integer.toString(iter) + "| ");
			try {
				Test.test();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			
			for (int u : Data.trainUserNo) {
				Data.client[u].setIv(new LinkedList<Integer>(Data.I_u[u]));
			}
			
			int rating_num = 0;
			while(rating_num < Data.rating_num) {
				
				for (int u : Data.trainUserNo) {
					Data.client[u].setGrad_neighborhood_V(new float[Data.m+1][Data.d]);
					Data.client[u].setItemNum(new HashMap<Integer, Integer>());
				}

				for (int u=1; u<Data.n+1; u++) {


					// --- Calculate gradient via rated items 
					if(Data.client[u].getIu().size() != 0)
					{

						int index = (int)(Math.random()*Data.client[u].getIu().size());

						int i = Data.client[u].getIu().remove(index);
						rating_num++;
						//					System.out.println("i = " + i);
						float pred = 0; 
						for (int f=0; f<Data.d; f++)
						{
							pred += Data.U[u][f] * Data.client[u].getV()[i][f];
						}
						float error = (float) (Data.traningDataMap.get(u).get(i) - pred);

						float grad_U[] = new float[Data.d];
						float grad_V[] = new float[Data.d];
						for(int f=0; f<Data.d; f++)
						{	
							// --- Calculate the gradients of user and item
							grad_U[f] = -error * Data.client[u].getV()[i][f] + Data.lambda * Data.U[u][f];
							grad_V[f] = -error * Data.U[u][f] + Data.lambda * Data.client[u].getV()[i][f];

							// --- Update user and item specific
							// latent feature locally
							Data.U[u][f] = (float) (Data.U[u][f] - Data.client[u].getGamma() * grad_U[f]);
							Data.client[u].getV()[i][f] = (float) (Data.client[u].getV()[i][f] - Data.client[u].getGamma() * grad_V[f]);
						}

						if(Data.neighborhoodHashMap.containsKey(u))
						{
							for (Integer neighbor : Data.neighborhoodHashMap.get(u)) {

								Data.client[neighbor].receiveGradientFromNeighbor(i, Data.userSimilarityMatrix[u][neighbor], grad_V);

							}
						}
					}
				}

				for (int u=1; u<Data.n+1; u++) {
					// --- update Vi after all clients have exchanged their item gradients
					if(Data.neighborhoodHashMap.containsKey(u))
					{
						Data.client[u].updateV();
					}
				}
			}

			for (int u=1; u<Data.n+1; u++) {
				Data.client[u].updatGamma(); 
			}
		}

	}
} 