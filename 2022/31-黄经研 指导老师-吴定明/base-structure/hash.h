#include<iostream>
#include<math.h>
#include<string.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t; 
typedef unsigned int uint32_t;
#define MAXNUM 1000000
#define BIT_SIZE 16
#include <iostream>
#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif 

using namespace std;
unsigned int 
BOB4(const unsigned char * str, unsigned int len);

unsigned int 
BOB1(const unsigned char * str, unsigned int len);

unsigned int 
BOB2(const unsigned char * str, unsigned int len);

unsigned int 
BOB3(const unsigned char * str, unsigned int len);

unsigned int 
PJW(const unsigned char * str, unsigned int len);

unsigned int 
SDBW(const unsigned char * str, unsigned int len);

unsigned int
DJB(const unsigned char * str, unsigned int len);

unsigned int 
RS(const unsigned char * str, unsigned int len);

unsigned int 
JS(const unsigned char * str, unsigned int len);

unsigned int 
Hsieh(const unsigned char* str, unsigned int len);

unsigned int
RSHash(const unsigned char* str, unsigned int len);

unsigned int
JSHash(const unsigned char* str, unsigned int len);

unsigned int
BKDR(const unsigned char* str, unsigned int len);

unsigned int 
DJBHash(const unsigned char* str, unsigned int len);

unsigned  int
DEKHash(const unsigned char* str, unsigned int len);

unsigned int 
APHash(const unsigned char* str, unsigned int len);

unsigned int
CRC32(const unsigned char* str, unsigned int len);

unsigned int
SDBM(const unsigned char *str, unsigned int len);

unsigned int
OCaml(const unsigned char *str, unsigned int len);

unsigned int
SML(const unsigned char *str, unsigned int len);

unsigned int
STL(const unsigned char *str, unsigned int len);

unsigned int
FNV32(const unsigned char *str, unsigned int len);

unsigned int
PJWHash (const unsigned char *str, unsigned int len);


unsigned int 
Hsieh (const unsigned char * str, unsigned int len)
{
	unsigned int hash = len, tmp;
	int rem;

    //if (len <= 0 || str == NULL) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (str);
        tmp    = (get16bits (str+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        str  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (str);
                hash ^= hash << 16;
                hash ^= ((signed char)str[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (str);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*str;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}

unsigned int 
RSHash(const unsigned char * str, unsigned int len)
{
    unsigned int b = 378551;
    unsigned int a = 63689;
    unsigned int hash = 0;
	
	for(int i = 0; i < len; i++){
		hash = hash * a + (*str++);
        a *= b;
	}
    //return (hash & 0x7FFFFFFF);
	return hash;
}

unsigned int 
JSHash(const unsigned char * str, unsigned int len)
{
	unsigned int hash = 1315423911;  
	for(int i = 0; i < len; i++)  
	{  
		hash ^= ((hash << 5) + str[i] + (hash >> 2));  
	}  
	return hash;
}

unsigned int 
BKDR(const unsigned char * str, unsigned int len)
{	
	unsigned int seed = 131; // 31 131 1313 13131 131313 etc..  
    unsigned int hash = 0;  
	for(int i = 0; i < len; i++)  
	{  
		hash = (hash * seed) + str[i]; 
	}
	
	return hash;
}


unsigned int 
DEKHash(const unsigned char * str, unsigned int len)   
{   
    
    unsigned int hash = 1315423911;
	for(int i = 0; i < len; i++){
		hash = ((hash << 5) ^ (hash >> 27)) ^ str[i];
	}
	
    return hash;
}

// AP Hash 
unsigned int 
APHash(const unsigned char *str, unsigned int len)
{
	unsigned int hash = 0;
	
 
	for (int i = 0; i < len; i++)
	{
		if ((i & 1) == 0)
		{
			hash ^= ((hash << 7) ^ (*str++) ^ (hash >> 3));
		}
		else
		{
			hash ^= (~((hash << 11) ^ (*str++) ^ (hash >> 5)));
		}
	}
 
	return (hash & 0x7FFFFFFF);
//	return hash;
}

unsigned int 
CRC32(const unsigned char * str,unsigned int len)
{  
	unsigned int crctab[256] =
	{
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3, 
    0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 
    0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7, 
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5, 
    0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 
    0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59, 
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F, 
    0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 
    0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433, 
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01, 
    0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 
    0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65, 
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 
    0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 
    0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F, 
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD, 
    0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 
    0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1, 
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7, 
    0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 
    0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B, 
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79, 
    0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 
    0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D, 
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713, 
    0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 
    0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777, 
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45, 
    0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 
    0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9, 
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF, 
    0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
	};
    unsigned int hash;     

    hash=0xffffffff;
    for(int i = 0; i < len; i++){
        hash = (hash >> 8) ^ crctab[(hash & 0xFF) ^ str[i]];     
    }  
    
    hash ^= 0xFFFFFFFF;  
    return hash;     
}

unsigned int
SDBM(const unsigned char *str, unsigned int len)
{
	unsigned int hash = 0;
	
	for (int i=0; i<len; i++) {
		hash = str[i] + (hash << 6) + (hash << 16) - hash;
	}
	
	return hash;
}

unsigned int
OCaml(const unsigned char *str, unsigned int len) 
{
	unsigned int hash = 0;

	for (int i=0; i<len; i++) {
		hash = hash*19 + str[i];
	}

	return hash;
}

unsigned int
SML(const unsigned char *str, unsigned int len)
{
	unsigned int hash = 0;

	for (int i=0; i<len; i++) 
	{
		hash = 33*hash + 720 + str[i];
	}

	return hash;
}

unsigned int
STL(const unsigned char *str, unsigned int len) 
{
	unsigned int hash = 0;

	for (int i=0; i<len; i++) 
	{
		hash = 5*hash + str[i];
	}

	return hash;
}



unsigned int
FNV32(const unsigned char *str, unsigned int len)
{
    //unsigned char *bp = (unsigned char *)str;	/* start of buffer */
    //unsigned char *be = str + len;		/* beyond end of buffer */
	unsigned int hval = 0;
    for(int i = 0; i < len; i++) {
		hval *= 0x01000193;

		//hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
		
		/* xor the bottom with the current octet */
		hval ^= (unsigned int)*str++;
    }

    /* return our new hash value */
    return hval;
}

// P. J. Weinberger Hash Function
unsigned int
PJWHash (const unsigned char *str, unsigned int len)
{
    unsigned int BitsInUnignedInt = (unsigned int) (sizeof (unsigned
							    int) * 8);
    unsigned int ThreeQuarters = (unsigned int) ((BitsInUnignedInt * 3) / 4);
    unsigned int OneEighth = (unsigned int) (BitsInUnignedInt / 8);
    unsigned int HighBits = (unsigned int) (0xFFFFFFFF) <<
	(BitsInUnignedInt - OneEighth);
    unsigned int hash = 0;
    unsigned int test = 0;
    for(int i = 0; i < len; i++)
    {
		hash = (hash << OneEighth) + (*str++);
		if ((test = hash & HighBits) != 0)
		{
			hash = ((hash ^ (test >> ThreeQuarters)) & (~HighBits));
		}
	}
    return hash ;
}
#define mix(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>13); \
  b -= c; b -= a; b ^= (a<<8); \
  c -= a; c -= b; c ^= (b>>13); \
  a -= b; a -= c; a ^= (c>>12);  \
  b -= c; b -= a; b ^= (a<<16); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>3);  \
  b -= c; b -= a; b ^= (a<<10); \
  c -= a; c -= b; c ^= (b>>15); \
}

unsigned int 
BOB4(const unsigned char * str, unsigned int len)
{
	//register ub4 a,b,c,len;
	unsigned int a,b,c;
	unsigned int initval = 127;
	/* Set up the internal state */
	//len = length;
	a = b = 0x9e3779b9;  /* the golden ratio; an arbitrary value */
	c = initval;         /* the previous hash value */

	/*---------------------------------------- handle most of the key */
	while (len >= 12)
	{
		a += (str[0] +((unsigned int)str[1]<<8) +((unsigned int)str[2]<<16) +((unsigned int)str[3]<<24));
		b += (str[4] +((unsigned int)str[5]<<8) +((unsigned int)str[6]<<16) +((unsigned int)str[7]<<24));
		c += (str[8] +((unsigned int)str[9]<<8) +((unsigned int)str[10]<<16)+((unsigned int)str[11]<<24));
		mix(a,b,c);
		str += 12; len -= 12;
	}

	/*------------------------------------- handle the last 11 bytes */
	c += len;
	switch(len)              /* all the case statements fall through */
	{
		case 11: c+=((unsigned int)str[10]<<24);
		case 10: c+=((unsigned int)str[9]<<16);
		case 9 : c+=((unsigned int)str[8]<<8);
		/* the first byte of c is reserved for the length */
		case 8 : b+=((unsigned int)str[7]<<24);
		case 7 : b+=((unsigned int)str[6]<<16);
		case 6 : b+=((unsigned int)str[5]<<8);
		case 5 : b+=str[4];
		case 4 : a+=((unsigned int)str[3]<<24);
		case 3 : a+=((unsigned int)str[2]<<16);
		case 2 : a+=((unsigned int)str[1]<<8);
		case 1 : a+=str[0];
		/* case 0: nothing left to add */
	}
	mix(a,b,c);
	/*-------------------------------------------- report the result */
	return c;
}

unsigned int 
BOB1(const unsigned char * str, unsigned int len)
{
	//register ub4 a,b,c,len;
	unsigned int a,b,c;
	unsigned int initval = 2;
	/* Set up the internal state */
	//len = length;
	a = b = 0x9e3779b9;  /* the golden ratio; an arbitrary value */
	c = initval;         /* the previous hash value */

	/*---------------------------------------- handle most of the key */
	while (len >= 12)
	{
		a += (str[0] +((unsigned int)str[1]<<8) +((unsigned int)str[2]<<16) +((unsigned int)str[3]<<24));
		b += (str[4] +((unsigned int)str[5]<<8) +((unsigned int)str[6]<<16) +((unsigned int)str[7]<<24));
		c += (str[8] +((unsigned int)str[9]<<8) +((unsigned int)str[10]<<16)+((unsigned int)str[11]<<24));
		mix(a,b,c);
		str += 12; len -= 12;
	}

	/*------------------------------------- handle the last 11 bytes */
	c += len;
	switch(len)              /* all the case statements fall through */
	{
		case 11: c+=((unsigned int)str[10]<<24);
		case 10: c+=((unsigned int)str[9]<<16);
		case 9 : c+=((unsigned int)str[8]<<8);
		/* the first byte of c is reserved for the length */
		case 8 : b+=((unsigned int)str[7]<<24);
		case 7 : b+=((unsigned int)str[6]<<16);
		case 6 : b+=((unsigned int)str[5]<<8);
		case 5 : b+=str[4];
		case 4 : a+=((unsigned int)str[3]<<24);
		case 3 : a+=((unsigned int)str[2]<<16);
		case 2 : a+=((unsigned int)str[1]<<8);
		case 1 : a+=str[0];
		/* case 0: nothing left to add */
	}
	mix(a,b,c);
	/*-------------------------------------------- report the result */
	return c;
}
unsigned int 
BOB2(const unsigned char * str, unsigned int len)
{
	//register ub4 a,b,c,len;
	unsigned int a,b,c;
	unsigned int initval = 31;
	/* Set up the internal state */
	//len = length;
	a = b = 0x9e3779b9;  /* the golden ratio; an arbitrary value */
	c = initval;         /* the previous hash value */

	/*---------------------------------------- handle most of the key */
	while (len >= 12)
	{
		a += (str[0] +((unsigned int)str[1]<<8) +((unsigned int)str[2]<<16) +((unsigned int)str[3]<<24));
		b += (str[4] +((unsigned int)str[5]<<8) +((unsigned int)str[6]<<16) +((unsigned int)str[7]<<24));
		c += (str[8] +((unsigned int)str[9]<<8) +((unsigned int)str[10]<<16)+((unsigned int)str[11]<<24));
		mix(a,b,c);
		str += 12; len -= 12;
	}

	/*------------------------------------- handle the last 11 bytes */
	c += len;
	switch(len)              /* all the case statements fall through */
	{
		case 11: c+=((unsigned int)str[10]<<24);
		case 10: c+=((unsigned int)str[9]<<16);
		case 9 : c+=((unsigned int)str[8]<<8);
		/* the first byte of c is reserved for the length */
		case 8 : b+=((unsigned int)str[7]<<24);
		case 7 : b+=((unsigned int)str[6]<<16);
		case 6 : b+=((unsigned int)str[5]<<8);
		case 5 : b+=str[4];
		case 4 : a+=((unsigned int)str[3]<<24);
		case 3 : a+=((unsigned int)str[2]<<16);
		case 2 : a+=((unsigned int)str[1]<<8);
		case 1 : a+=str[0];
		/* case 0: nothing left to add */
	}
	mix(a,b,c);
	/*-------------------------------------------- report the result */
	return c;
}
unsigned int 
BOB3(const unsigned char * str, unsigned int len)
{
	//register ub4 a,b,c,len;
	unsigned int a,b,c;
	unsigned int initval = 73;
	/* Set up the internal state */
	//len = length;
	a = b = 0x9e3779b9;  /* the golden ratio; an arbitrary value */
	c = initval;         /* the previous hash value */

	/*---------------------------------------- handle most of the key */
	while (len >= 12)
	{
		a += (str[0] +((unsigned int)str[1]<<8) +((unsigned int)str[2]<<16) +((unsigned int)str[3]<<24));
		b += (str[4] +((unsigned int)str[5]<<8) +((unsigned int)str[6]<<16) +((unsigned int)str[7]<<24));
		c += (str[8] +((unsigned int)str[9]<<8) +((unsigned int)str[10]<<16)+((unsigned int)str[11]<<24));
		mix(a,b,c);
		str += 12; len -= 12;
	}

	/*------------------------------------- handle the last 11 bytes */
	c += len;
	switch(len)              /* all the case statements fall through */
	{
		case 11: c+=((unsigned int)str[10]<<24);
		case 10: c+=((unsigned int)str[9]<<16);
		case 9 : c+=((unsigned int)str[8]<<8);
		/* the first byte of c is reserved for the length */
		case 8 : b+=((unsigned int)str[7]<<24);
		case 7 : b+=((unsigned int)str[6]<<16);
		case 6 : b+=((unsigned int)str[5]<<8);
		case 5 : b+=str[4];
		case 4 : a+=((unsigned int)str[3]<<24);
		case 3 : a+=((unsigned int)str[2]<<16);
		case 2 : a+=((unsigned int)str[1]<<8);
		case 1 : a+=str[0];
		/* case 0: nothing left to add */
	}
	mix(a,b,c);
	/*-------------------------------------------- report the result */
	return c;
}
unsigned int RS(const unsigned char * str, unsigned int len)
{ 
      int b     = 378551;  
      int a     = 63689; 
      unsigned int hash = 0;  
      for(int i = 0; i < len; i++)  
      {  
         hash = hash * a + str[i];  
         a    = a * b;  
      }  
      return (hash & 0x7FFFFFFF); 
}
unsigned int JS(const unsigned char * str, unsigned int len)
{
	  long hash = 1315423911;  
      for(int i = 0; i < len; i++)  
      {  
         hash ^= ((hash << 5) + str[i] + (hash >> 2));  
      }  
      return hash;  
}
unsigned int PJW(const unsigned char * str, unsigned int len)
{      
	  long BitsInUnsignedInt = (long)(4 * 8);  
      long ThreeQuarters     = (long)((BitsInUnsignedInt  * 3) / 4);  
      long OneEighth         = (long)(BitsInUnsignedInt / 8);  
      long HighBits          = (long)(0xFFFFFFFF) << (BitsInUnsignedInt - OneEighth);  
      long hash              = 0;  
      long test              = 0;  
      for(int i = 0; i < len; i++)  
      {  
         hash = (hash << OneEighth) + str[i];  
         if((test = hash & HighBits)  != 0)  
         {  
            hash = (( hash ^ (test >> ThreeQuarters)) & (~HighBits));  
         }  
      }  
      return (hash & 0x7FFFFFFF);   
}
unsigned int SDBW(const unsigned char * str, unsigned int len)
{
	  long hash = 0;  
      for(int i = 0; i < len; i++)  
      {  
         hash = str[i] + (hash << 6) + (hash << 16) - hash;  
      }  
      return hash;  
}
unsigned int DJB(const unsigned char * str,unsigned int len)
{
	  long hash = 5381;  
      for(int i = 0; i < len; i++)  
      {  
         hash = ((hash << 5) + hash) + str[i];  
      }
	  hash &= ~(1 << 31);  
      return hash;
}
unsigned long long MurmurHash64B ( const void * key, int len, unsigned int seed=0xEE6B27EB )  
    {  
        const unsigned int m = 0x5bd1e995;  
        const int r = 24;  
       
        unsigned int h1 = seed ^ len;  
        unsigned int h2 = 0;  
       
        const unsigned int * data = (const unsigned int *)key;  
       
        while(len >= 8)  
        {  
            unsigned int k1 = *data++;  
            k1 *= m; k1 ^= k1 >> r; k1 *= m;  
            h1 *= m; h1 ^= k1;  
            len -= 4;  
       
            unsigned int k2 = *data++;  
            k2 *= m; k2 ^= k2 >> r; k2 *= m;  
            h2 *= m; h2 ^= k2;  
            len -= 4;  
        }  
        //cout<<"it is in"<<endl;
        if(len >= 4)  
        {  
            unsigned int k1 = *data++;  
            k1 *= m; k1 ^= k1 >> r; k1 *= m;  
            h1 *= m; h1 ^= k1;  
            len -= 4;  
        }  
       
        switch(len)  
        {  
        case 3: h2 ^= ((unsigned char*)data)[2] << 16;  
        case 2: h2 ^= ((unsigned char*)data)[1] << 8;  
        case 1: h2 ^= ((unsigned char*)data)[0];  
                h2 *= m;  
        };  
       
        h1 ^= h2 >> 18; h1 *= m;  
        h2 ^= h1 >> 22; h2 *= m;  
        h1 ^= h2 >> 17; h1 *= m;  
        h2 ^= h1 >> 19; h2 *= m;  
       
        unsigned long long h = h1;  
       
        h = (h << 32) | h2;  
       
        return h;  
    }
	   
    unsigned int MurmurHash(const unsigned char *key, unsigned int len)  
        {  
            const unsigned int m = 0x5bd1e995;  
            const int r = 24;  
            const int seed = 97;  
            unsigned int h = seed ^ len;  
            // Mix 4 bytes at a time into the hash  
            const unsigned char *data = (const unsigned char *)key;  
            while(len >= 4)  
            {  
                unsigned int k = *(unsigned int *)data;  
                k *= m;   
                k ^= k >> r;   
                k *= m;   
                h *= m;   
                h ^= k;  
                data += 4;  
                len -= 4;  
            }  
            // Handle the last few bytes of the input array  
            switch(len)  
            {  
                case 3: h ^= data[2] << 16;  
                case 2: h ^= data[1] << 8;  
                case 1: h ^= data[0];  
                h *= m;  
            };  
            // Do a few final mixes of the hash to ensure the last few  
            // bytes are well-incorporated.  
            h ^= h >> 13;  
            h *= m;  
            h ^= h >> 15;  
            return h;  
        } 
		unsigned int MurmurHash2(const unsigned char *key, unsigned int len)  
        {  
            const unsigned int m = 1315423911;  
            const int r = 24;  
            const int seed = 97;  
            unsigned int h = seed ^ len;  
            // Mix 4 bytes at a time into the hash  
            const unsigned char *data = (const unsigned char *)key;  
            while(len >= 4)  
            {  
                unsigned int k = *(unsigned int *)data;  
                k *= m;   
                k ^= k >> r;   
                k *= m;   
                h *= m;   
                h ^= k;  
                data += 4;  
                len -= 4;  
            }  
            // Handle the last few bytes of the input array  
            switch(len)  
            {  
                case 3: h ^= data[2] << 16;  
                case 2: h ^= data[1] << 8;  
                case 1: h ^= data[0];  
                h *= m;  
            };  
            // Do a few final mixes of the hash to ensure the last few  
            // bytes are well-incorporated.  
            h ^= h >> 13;  
            h *= m;  
            h ^= h >> 15;  
            return h;  
        } 
		unsigned int MurmurHash3(const unsigned char *key, unsigned int len)  
        {  
            const unsigned int m = 378551;  
            const int r = 24;  
            const int seed = 97;  
            unsigned int h = seed ^ len;  
            // Mix 4 bytes at a time into the hash  
            const unsigned char *data = (const unsigned char *)key;  
            while(len >= 4)  
            {  
                unsigned int k = *(unsigned int *)data;  
                k *= m;   
                k ^= k >> r;   
                k *= m;   
                h *= m;   
                h ^= k;  
                data += 4;  
                len -= 4;  
            }  
            // Handle the last few bytes of the input array  
            switch(len)  
            {  
                case 3: h ^= data[2] << 16;  
                case 2: h ^= data[1] << 8;  
                case 1: h ^= data[0];  
                h *= m;  
            };  
            // Do a few final mixes of the hash to ensure the last few  
            // bytes are well-incorporated.  
            h ^= h >> 13;  
            h *= m;  
            h ^= h >> 15;  
            return h;  
        } 
        unsigned int MurmurHash4(const unsigned char *key, unsigned int len)  
        {  
            const unsigned int m = 5381;  
            const int r = 24;  
            const int seed = 97;  
            unsigned int h = seed ^ len;  
            // Mix 4 bytes at a time into the hash  
            const unsigned char *data = (const unsigned char *)key;  
            while(len >= 4)  
            {  
                unsigned int k = *(unsigned int *)data;  
                k *= m;   
                k ^= k >> r;   
                k *= m;   
                h *= m;   
                h ^= k;  
                data += 4;  
                len -= 4;  
            }  
            // Handle the last few bytes of the input array  
            switch(len)  
            {  
                case 3: h ^= data[2] << 16;  
                case 2: h ^= data[1] << 8;  
                case 1: h ^= data[0];  
                h *= m;  
            };  
            // Do a few final mixes of the hash to ensure the last few  
            // bytes are well-incorporated.  
            h ^= h >> 13;  
            h *= m;  
            h ^= h >> 15;  
            return h;  
        } 
        unsigned int MurmurHash5(const unsigned char *key, unsigned int len)  
        {  
            const unsigned int m = 13627;  
            const int r = 24;  
            const int seed = 97;  
            unsigned int h = seed ^ len;  
            // Mix 4 bytes at a time into the hash  
            const unsigned char *data = (const unsigned char *)key;  
            while(len >= 4)  
            {  
                unsigned int k = *(unsigned int *)data;  
                k *= m;   
                k ^= k >> r;   
                k *= m;   
                h *= m;   
                h ^= k;  
                data += 4;  
                len -= 4;  
            }  
            // Handle the last few bytes of the input array  
            switch(len)  
            {  
                case 3: h ^= data[2] << 16;  
                case 2: h ^= data[1] << 8;  
                case 1: h ^= data[0];  
                h *= m;  
            };  
            // Do a few final mixes of the hash to ensure the last few  
            // bytes are well-incorporated.  
            h ^= h >> 13;  
            h *= m;  
            h ^= h >> 15;  
            return h;  
        } 

unsigned int (*hashfun[14])(const unsigned char*, unsigned int)=
{
    MurmurHash,
    BOB1,
    MurmurHash4,
	BOB2,
	CRC32,
    BOB3,
    BOB4,
    PJW,
    DJB,
    OCaml,
    SML,
  //  STL,
    FNV32,
    Hsieh,
    RSHash	
};		   

