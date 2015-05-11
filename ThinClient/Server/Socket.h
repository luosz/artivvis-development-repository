#ifndef SOCKET_H
#define SOCKET_H

#include <iostream>
#include <WinSock2.h>
#include "NetworkingMisc.h"
#include "Packet.h"

class Socket
{
public:
	int handle;

	virtual bool Open(unsigned short port) = 0;
	void Close();
};


#endif