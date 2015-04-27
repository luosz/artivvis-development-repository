#include "UDPSocket.h"

bool UDPSocket::Open(unsigned short port)
{
	handle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

	if (!Socket::Open(port))
		return false;

	DWORD nonBlocking = 1;
	if ( ioctlsocket(handle, FIONBIO, &nonBlocking) != 0 )
	{
		std::cout << "Failed to set non blocking" << std::endl;
		return false;
	}

	return true;
}


bool UDPSocket::Send(IPAddress &dest, Packet &packet)
{
	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(dest.address);
	addr.sin_port = htons(dest.udpPort);

	int numSentBytes = sendto(handle, (const char*)packet.message, packet.size, 0, (sockaddr*)&addr, sizeof(sockaddr_in));

	if (numSentBytes != packet.size)
	{
		std::cout << "Failed to send packet: " << packet.size << std::endl;
		return false;
	}

	return true;
}

bool UDPSocket::Receive(IPAddress &sender, Packet &packet)
{
	sockaddr_in fromAddr;	
	int fromLength = sizeof(fromAddr);

	packet.size = recvfrom(handle, (char*)packet.message, MAX_PACKET_SIZE, 0, (sockaddr*)&fromAddr, &fromLength);

	if (packet.size > 0)
	{
		sender.address = ntohl(fromAddr.sin_addr.s_addr);
		sender.udpPort = ntohs(fromAddr.sin_port);

		packet.type = PacketType(packet.ReadByte());

		return true;
	}
	else
		return false;
}