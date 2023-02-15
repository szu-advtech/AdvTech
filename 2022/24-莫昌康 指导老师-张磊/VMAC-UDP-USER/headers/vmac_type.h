#include <stdio.h>
#include <stdlib.h>

#ifndef _SIZE_T
#define _SIZE_T
typedef unsigned long size_t;
#endif

typedef unsigned char u8;

struct kvec
{
  void *iov_base;
  size_t iov_len;
};

struct iov_iter
{
  u8 iter_type;
  _Bool datasource;
  size_t iov_offset;
  size_t count;
  union
  {
    const struct iovec *iov;
    const struct kvec *kvec;
    // const struct bio_vec *bvec;
    // struct xarray *xarray;
    // struct pipe_inode_info *pipe;
  };
  union
  {
    unsigned long nr_segs;
    struct
    {
      unsigned int head;
      unsigned int start_head;
    };
    long xarray_start;
  };
};

struct msghdr_vmac
{
  void *msg_name;        /* Address to send to/receive from.  */
  socklen_t msg_namelen; /* Length of address data.  */

  struct iovec *msg_iov; /* Vector of data to send/receive into.  */
  size_t msg_iovlen;     /* Number of elements in the vector.  */

  void *msg_control;     /* Ancillary data (eg BSD filedesc passing). */
  size_t msg_controllen; /* Ancillary data buffer length.
        !! The type should be socklen_t but the
        definition of the kernel is incompatible
        with this.  */

  int msg_flags; /* Flags on received message.  */

  struct iov_iter msg_iter; /* data */
};