
MJPG_STREAMER_PLUGIN_OPTION(output_udp "UDP output stream plugin")
MJPG_STREAMER_PLUGIN_COMPILE(output_udp output_udp.c vmac-usrsp.c uthash.h vmac-usrsp.h csiphash.c)
