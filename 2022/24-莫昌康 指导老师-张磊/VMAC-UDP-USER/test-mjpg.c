#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define BUFLEN 1024 // Max length of buffer

int main()
{
  int video_sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  struct sockaddr_in video_addr;
  memset(&video_addr, 0, sizeof(video_addr));
  video_addr.sin_family = AF_INET;
  video_addr.sin_port = htons(8080);
  video_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

  connect(video_sockfd, (struct sockaddr *)&video_addr, sizeof(video_addr));

  printf("connect to mjpg-streamer success\n");

  char *buf = (char *)malloc(102400);
  memset(buf, 0, 102400);
  strcpy(buf, "GET /?action=stream\n");
  send(video_sockfd, buf, strlen(buf), 0);
  send(video_sockfd, "f\n", 2, 0);

  memset(buf, 0, 102400);
  recv(video_sockfd, buf, BUFLEN, 0);

  int recv_size, pic_length = 0, p = 0;
  char *begin, *end;
  char cont_len[10] = {0};
  char *pic_data = (char *)malloc(102400);

  while (1)
  {
    memset(buf, 0, 102400);
    recv_size = recv(video_sockfd, buf, 74, 0);
    if (strstr(buf, "Content-Type"))
    {
      begin = strstr(buf, "Content-Length");
      end = strstr(buf, "X-Timestamp");
      memcpy(cont_len, begin + 16, end - 2 - begin - 16);
      pic_length = atoi(cont_len);
      printf("recv head Content-Length = %d %d\n", atoi(cont_len), recv_size);
      memset(cont_len, 0, 10);
    }
    else
    {
      continue;
    }

    while (1)
    {
      memset(buf, 0, 102400);
      recv_size = recv(video_sockfd, buf, pic_length, 0);
      if (recv_size == pic_length)
      {
        memcpy(pic_data + p, buf, recv_size);
        p += recv_size;
        //处理图片数据
        p = 0;
        memset(pic_data, 0, 102400);
        pic_length = 0;
        break;
      }
      else
      {
        memcpy(pic_data + p, buf, recv_size);
        pic_length = pic_length - recv_size;
        p += recv_size;
      }
    }

    recv(video_sockfd, buf, 24, 0);
  }
}