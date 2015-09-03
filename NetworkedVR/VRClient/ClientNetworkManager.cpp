#include "ClientNetworkManager.h"

NetworkManager::NetworkManager()
{
	// Binding sockets to zero auto detects an open port
	udpPort = 0;
	tcpPort = 0;
	connected = false;
}

void NetworkManager::Init(VolumeRenderer *renderer_)
{
	renderer = renderer_;
	volume = &renderer->volume;

	WSADATA wsaData;
	if ( WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR )
		std::cout << "Failed to Initialize Sockets" << std::endl;

	if (!udpSocket.Open(udpPort))
		std::cout << "Failed to Open Socket" << std::endl;

	if (!tcpSocket.Open(tcpPort))
		std::cout << "Failed to Open Socket" << std::endl;

	sockaddr_in addr;
	int len = sizeof(addr);
	getsockname(udpSocket.handle, (sockaddr*)&addr, &len);
	udpPort = ntohs(addr.sin_port);

	server = IPAddress(127, 0, 0, 1, 40000);
//	server = IPAddress(134, 226, 54, 9, 40000);
	server.tcpPort = 40001;

	InitTCPMsgHandler();


	bool serverFound = false;

	while(!serverFound)
		serverFound = LogIn();

//	asyncListener = std::async(&NetworkManager::AsyncCheckForMessages, this);
}


bool NetworkManager::InitTCPMsgHandler()
{
	DWORD w = WSAGetLastError();

	WNDCLASSEX wClass = {};

	wClass.cbSize = sizeof(WNDCLASSEX);
	wClass.lpfnWndProc = (WNDPROC)MsgWndProc;
	wClass.lpszClassName = _T("Client Window Class");

	if ( !RegisterClassEx(&wClass) )
	{
		std::cout << "Register msgWnd class failed" << std::endl;
		return false;
	}
	
	msgWnd = CreateWindowEx(0, _T("Client Window Class"), _T("Client Titlebar"), WS_OVERLAPPEDWINDOW, CW_DEFAULT, CW_DEFAULT, CW_DEFAULT, CW_DEFAULT, HWND_MESSAGE, (HMENU)NULL, (HINSTANCE)NULL, (LPVOID)NULL);

	if (msgWnd == NULL) 
	{
		std::cout << "Failed to create window" << std::endl;
		return false;
	}

	SetWindowLongPtr(msgWnd, GWLP_USERDATA, (LONG)this);

	return true;
}


bool NetworkManager::LogIn()
{
	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(server.address);
	addr.sin_port = htons(server.tcpPort);

	std::cout << "Connecting to " << (int)server.A() << "." << (int)server.B() << "." << (int)server.C() << "." << (int)server.D() << ":" << server.tcpPort << std::endl;

	if(connect(tcpSocket.handle, (sockaddr*)&addr, sizeof(sockaddr_in)) == SOCKET_ERROR)
	{
		std::cout << "Error connecting: " << WSAGetLastError() << std::endl;
		return false;
	}
	else
	{
		if (WSAAsyncSelect(tcpSocket.handle, msgWnd, NOTIFICATION_NUMBER, (FD_ACCEPT | FD_READ | FD_CLOSE)) != 0)
		{
			printf("Failed to init async socket: (%d)[1]\n", WSAGetLastError());
			return false;
		}


		std::cout << "Connected" << std::endl;

		Packet packet;
		packet.WriteByte((unsigned char)PacketType::INITIALIZATION);
		packet.WriteShort((unsigned short)udpPort);


		if (tcpSocket.Send(packet))
			std::cout << "Sent login" << std::endl;

		return true;
	}
}

void NetworkManager::AsyncCheckForMessages()
{
//	while (true)
//		CheckForMessages();

	for (int i=0; i<100000; i++)
		CheckForMessages();
}

void NetworkManager::CheckForMessages()
{
	int newPackSize = recv(tcpSocket.handle, (char*)(tcpPacket.message + tcpPacket.size), MAX_PACKET_SIZE - tcpPacket.size, 0);

//	std::cout << newPackSize << std::endl;

	if (newPackSize <= 0)
		return;

	tcpPacket.size += newPackSize;

	if (tcpPacket.size <= 0)
		return;

	int amountRead = 0;
	int chunkSize = 0;

	while (amountRead < tcpPacket.size)
	{
		chunkSize = tcpPacket.ReadInt();

		int bytesRemaining = tcpPacket.size - amountRead;

		// Checks if a smaller packet has been split at the end of the bigger one and buffers the beginning of it to be complete on the next packet received
		if (chunkSize > bytesRemaining)
		{
			std::memcpy(tcpPacket.message, tcpPacket.message + amountRead, bytesRemaining);
			tcpPacket.size = bytesRemaining;
			tcpPacket.readPosition = 0;

			return;
		}

		tcpPacket.type = (PacketType)tcpPacket.ReadByte();

		switch (tcpPacket.type)
		{
			case PacketType::INITIALIZATION:
				std::cout << tcpPacket.size << " initialization packet received" << std::endl;
				ReceiveInitialization(tcpPacket);

				break;

			case PacketType::BLOCK:
//				std::cout << "Block packet size: " << chunkSize << std::endl;
				UpdateBlock(tcpPacket);
				break;

			default:
				std::cout << "Unknown packet type" << std::endl;
				tcpPacket.size = 0;
				tcpPacket.readPosition = 0;
				return;
		}

		amountRead += chunkSize;
	}

	tcpPacket.size = 0;
	tcpPacket.readPosition = 0;

	return;
}


void NetworkManager::ReceiveInitialization(Packet &packet)
{
	volume->timesteps = packet.ReadInt();
	volume->timePerFrame = packet.ReadFloat();
	volume->xRes = packet.ReadInt();
	volume->yRes = packet.ReadInt();
	volume->zRes = packet.ReadInt();
	volume->bytesPerElement = packet.ReadInt();
	volume->littleEndian = packet.ReadBool();

	volume->Init();
}

void NetworkManager::UpdateBlock(Packet &packet)
{
	int ID;

	int blockRes = packet.ReadInt();
	int blockX = packet.ReadInt();
	int blockY = packet.ReadInt();
	int blockZ = packet.ReadInt();

	int xMin = blockX * blockRes;
	int yMin = blockY * blockRes;
	int zMin = blockZ * blockRes;

//	std::cout << blockX << " - " << blockY << " - " << blockZ << std::endl;

	for (int z=0; z<blockRes; z++)
		for (int y=0; y<blockRes; y++)
			for (int x=0; x<blockRes; x++)
			{
				if ((xMin + x) >= volume->xRes || (yMin + y) >= volume->yRes || (zMin + z) >= volume->zRes)
					continue;

				ID = (xMin + x) + ((yMin + y) * volume->xRes) + ((zMin + z) * volume->xRes * volume->yRes);

				volume->memblock3D[ID] = packet.ReadByte();
			}
}



void NetworkManager::ReadMessage(WPARAM wParam)
{
	int newPackSize = recv(wParam, (char*)(tcpPacket.message + tcpPacket.size), MAX_PACKET_SIZE - tcpPacket.size, 0);

	tcpPacket.size += newPackSize;

	if (tcpPacket.size <= 0)
		return;

	int amountRead = 0;
	int chunkSize = 0;

	while (amountRead < tcpPacket.size)
	{
		chunkSize = tcpPacket.ReadInt();

		int bytesRemaining = tcpPacket.size - amountRead;

		// Checks if a smaller packet has been split at the end of the bigger one and buffers the beginning of it to be complete on the next packet received
		if (chunkSize > bytesRemaining)
		{
			std::memcpy(tcpPacket.message, tcpPacket.message + amountRead, bytesRemaining);
			tcpPacket.size = bytesRemaining;
			tcpPacket.readPosition = 0;

			return;
		}

		tcpPacket.type = (PacketType)tcpPacket.ReadByte();

		switch (tcpPacket.type)
		{
			case PacketType::INITIALIZATION:
				std::cout << tcpPacket.size << " initialization packet received" << std::endl;
				ReceiveInitialization(tcpPacket);

				break;

			case PacketType::BLOCK:
//				std::cout << "Block packet size: " << chunkSize << std::endl;
				UpdateBlock(tcpPacket);
				break;

			default:
				std::cout << "Unknown packet type" << std::endl;
				tcpPacket.size = 0;
				tcpPacket.readPosition = 0;
				return;
		}

		amountRead += chunkSize;
	}

	tcpPacket.size = 0;
	tcpPacket.readPosition = 0;
}

LRESULT CALLBACK  NetworkManager::ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (lParam)
	{
		case FD_ACCEPT:
		    break;

		case FD_READ:
//		    std::cout << "Incoming data: " << std::endl;
//			ReadMessage(wParam);
		    break;
		
		case FD_CLOSE:
			std::cout << "Connection Lost" << std::endl;
		    //Lost the connection
		    break;
	}

	return 0;
}


LRESULT CALLBACK NetworkManager::MsgWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	NetworkManager* netMgr = (NetworkManager*)GetWindowLongPtr(hwnd, GWLP_USERDATA);

    switch (message)
    {
		case NOTIFICATION_NUMBER:
			netMgr->ProcessMessage(hwnd, message, wParam, lParam);
		    break;

		default:
		    return DefWindowProc(hwnd, message, wParam, lParam);
    }

    return 0;
}


NetworkManager::~NetworkManager()
{
	udpSocket.Close();
	tcpSocket.Close();
}