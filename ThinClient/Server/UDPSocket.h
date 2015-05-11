#ifndef UDP_SOCKET_H
#define UDP_SOCKET_H

#include "Socket.h"

class UDPSocket		:		public Socket
{
public:
	bool Open(unsigned short port);

	bool Send(IPAddress &dest, Packet &packet);
	bool Receive(IPAddress &sender, Packet &packet);

	
};


#endif