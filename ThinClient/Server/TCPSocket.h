#ifndef TCP_SOCKET_H
#define TCP_SOCKET_H

#include "Socket.h"

class TCPSocket		:		public Socket
{
public:

	bool Open(unsigned short port);

	bool Send(Packet &packet);
	bool Receive(Packet &packet);
};


#endif