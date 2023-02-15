/*
Class with MPI_operation
*/

#include "ParSecureML_noMPI.h"
#include "mpi.h"

class Support: public Support_noMPI{
    public:
        void Send(int MPI_dest1, int MPI_dest2, MPI_Comm client2server1_comm, MPI_Comm client2server2_comm);
};

class Triplet: public Triplet_noMPI{
    public:
        void Rec(int MPI_dest, MPI_Comm server2server_comm);
        void Recv(int MPI_dest, MPI_Comm client2server_comm);
        void Activation(int MPI_dest, MPI_Comm server2server_comm);
};

class ConvSupport: public ConvSupport_noMPI{
	public:
		void Send(int MPI_dest1, int MPI_dest2, MPI_Comm client2server1_comm, MPI_Comm client2server2_comm);
};

class ConvTriplet: public ConvTriplet_noMPI{
	public:
	  	void Rec(int MPI_dest, MPI_Comm server2server_comm);
		void Recv(int MPI_dest, MPI_Comm client2server_comm);
};



