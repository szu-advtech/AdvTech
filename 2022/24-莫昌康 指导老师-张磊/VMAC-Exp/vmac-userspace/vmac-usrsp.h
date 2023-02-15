#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <linux/netlink.h>
#include <syslog.h>
#include <pthread.h>
#include <setjmp.h>
#include <sched.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include "uthash.h"

/** Defines **/

#define RATES_NUM 44

#define SGI 0x40
#define HT40 0x80
/* frame types */
#define VMAC_FC_INTEREST 0x00			/* Interest frame     */
#define VMAC_FC_DATA 0x01					/* Data frame 		  */
#define VMAC_FC_ANNOUNCEMENT 0x04 /* Announcement frame */
#define VMAC_FC_INJECT 0x05				/* Injected frame 	  */

/* netlink parameters */
#define VMAC_USER 0x1F		/* netlink ID to communicate with V-MAC Kernel Module */
#define MAX_PAYLOAD 0x7FF /* 2KB max payload per-frame */

/** Structs **/
const static struct
{
	double rate; /* Mbps */
	uint8_t rix; /* Index internally to pass to PHY */
} rates[RATES_NUM] =
		{
				{1.0, 0},
				{2.0, 1},
				{5.5, 2},
				{6.0, 4},
				{9.0, 5},
				{10.0, 7},
				{11.0, 3},
				{12.0, 6},
				{18.0, 8},
				{36.0, 9},
				{48.0, 10},
				{54.0, 11},
				{6.5, 12},
				{13.0, 13},
				{19.5, 14},
				{26.0, 15},
				{39.0, 16},
				{52.0, 18},
				{58.5, 20},
				{65.0, 22},
				{13.5, 12 + HT40},
				{27.0, 13 + HT40},
				{40.5, 14 + HT40},
				{54.0, 15 + HT40},
				{81.0, 16 + HT40},
				{108.0, 19 + HT40},
				{121.5, 20 + HT40},
				{135.0, 22 + HT40},
				{7.2, 12 + SGI},
				{14.4, 13 + SGI},
				{21.7, 14 + SGI},
				{28.9, 15 + SGI},
				{43.3, 16 + SGI},
				{57.8, 18 + SGI},
				{65.0, 20 + SGI},
				{72.2, 22 + SGI},
				{15.0, 12 + HT40 + SGI},
				{30.0, 13 + HT40 + SGI},
				{45.0, 14 + HT40 + SGI},
				{60.0, 15 + HT40 + SGI},
				{90.0, 16 + HT40 + SGI},
				{120.0, 18 + HT40 + SGI},
				{135.0, 20 + HT40 + SGI},
				{150.0, 22 + HT40 + SGI}};

/**
 * @brief      vmac frame information, buffer and interest name with their lengths, respectively.
 */
struct vmac_frame
{
	char *buf;
	uint16_t len;
	char *InterestName;
	uint16_t name_len;
};

/**
 * @brief      meta data to provide guidance to V-MAC configurations
 */
struct meta_data
{
	uint8_t type;
	uint16_t seq;
	double rate;
	uint64_t enc;
};

/**
 ** ABI Be careful when changing to adjust kernel/userspace information as well.
 **/
struct control
{
	char type[1];
	char rate[1];
	char enc[8];
	char seq[2];
	char bwsg;
	char rate_idx;
	char signal;
};

/* Struct to hash interest name to 64-bit encoding */
struct hash
{
	uint64_t id;
	char *name;
	UT_hash_handle hh;
};

/**
 * @brief      V-MAC userspace's library internal use configurations.
 */
struct vmac_lib_priv
{
	struct hash *names;
	struct sockaddr_nl src_addr, dest_addr;
	/* TX structs */
	struct nlmsghdr *nlh;
	struct iovec iov;
	struct msghdr msg;

	/* RX structs */
	struct nlmsghdr *nlh2;
	struct iovec iov2;
	struct msghdr msg2;

	uint64_t digest64;
	uint8_t fixed_rate;
	void (*cb)();
	char msgy[2000]; /* buffer to store frame */
	int sock_fd;
	pthread_t thread;
	char key[16];
};

extern struct vmac_lib_priv vmac_priv;

// package_info to provide guidance for file size,  可以整合到meta里面
struct package_info
{
	uint16_t pack_num; // 将文件分成pack_num份
};

/* Prototype functions */
uint64_t siphash24(const char *in, unsigned long inlen, const char k[16]);
int send_vmac(struct vmac_frame *frame, struct meta_data *meta);
void add_name(char *InterestName, uint16_t name_len);
void del_name(char *InterestName, uint16_t name_len);
int vmac_register(void (*cf)(struct vmac_frame *, struct meta_data *));