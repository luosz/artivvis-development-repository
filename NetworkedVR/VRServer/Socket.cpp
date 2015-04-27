#include "Socket.h"


bool Socket::Open(unsigned short port)
{
	if (handle <= 0)
	{
		std::cout << "Failed to create socket" << std::endl;
		return false;
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(port);

	if ( bind(handle, (const sockaddr*)&addr, sizeof(sockaddr_in)) < 0 )
	{
		std::cout << "Failed to bind socket" << std::endl;
		return false;
	}

	return true;
}



void Socket::Close()
{
	closesocket(handle);
}