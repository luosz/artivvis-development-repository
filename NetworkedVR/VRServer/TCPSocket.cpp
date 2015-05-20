#include "TCPSocket.h"

bool TCPSocket::Open(unsigned short port)
{
	handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	if (!Socket::Open(port))
		return false;

//	int flag = 1;
//	if ( setsockopt(handle, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) != 0 )
//	{
//		std::cout << "Failed to delay Nagle's algorithm" << std::endl;
//		return false;
//	}

	return true;
}


bool TCPSocket::Send(Packet &packet)
{
//	packet.WriteCheckSum();

	int numSentBytes = send(handle, (const char*)packet.message, packet.size, 0);

	if (numSentBytes != packet.size)
	{
//		std::cout << "Failed to send packet - error code " << WSAGetLastError() << std::endl;
		return false;
	}

	return true;
}


bool TCPSocket::Receive(Packet &packet)
{
	packet.size = recv(handle, (char*)packet.message, MAX_PACKET_SIZE, 0);

	if (packet.size > 0)
	{
//		int chksum = packet.ReadInt();
//		std::cout << "recv size: " << packet.size << "chksum: " << chksum << std::endl;

		packet.type = PacketType(packet.ReadByte());

		return true;
	}
	else
		return false;
}