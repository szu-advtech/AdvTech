1. local_path.py：
	The paths to the datasets and the result files.
2. read_data.py:
	Data processing file, read data set, process alphabetic data into numbers, process continuous numbers by data segment, store the processed results in ".pkl" file
3. Obfuscate_KMeans.py：
	Three obfuscate functions are random increase obfuscate, random flip obfuscate, obfuscate based on Kmeans. The obfuscate results are saved in the ".pkl" file.
4. evaluation_recommendation.py：
	In the absence of homomorphic encryption, the accuracy of recommendation is calculated by using the confused user-item interaction matrix.
	The accuracy of the recommendation is measured by RMSE.
5. evaluation_efficiency.py：
	Call the python file FedMF.py to calculate the elapsed time.
6. FedMF.py：
	Under the condition of using homomorphic encryption, only one round of interaction between the user and the server is carried out, and the time consumed by this round of interaction is calculated.
7. evaluation_attacks.py：
	Three models NB, SVM and GBDT were used to try the attack and calculate the accuracy of the attack.
8. seal_ckks.py:
	This is a kind of decorator which encapsulates the encryption and decryption functions of the homomorphic encryption algorithm.