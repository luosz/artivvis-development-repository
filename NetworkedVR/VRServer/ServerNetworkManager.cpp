#include "ServerNetworkManager.h"

void NetworkManager::Init(VolumeDataset &volume_)
{
	volume = &volume_;

	udpPort = 40000;
	tcpPort = 40001;

	WSADATA wsaData;
	if ( WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR )
		std::cout << "Failed to Initialize Sockets" << std::endl;

	if (!udpSocket.Open(udpPort))
		std::cout << "Failed to Open Socket" << std::endl;

	if (!tcpSocket.Open(tcpPort))
		std::cout << "Failed to Open Socket" << std::endl;

	InitTCPMsgHandler();
}


bool NetworkManager::InitTCPMsgHandler()
{
	WNDCLASSEX wClass = {};

	wClass.cbSize = sizeof(WNDCLASSEX);
	wClass.lpfnWndProc = (WNDPROC)MsgWndProc;
	wClass.lpszClassName = _T("Server Window Class");

	if ( !RegisterClassEx(&wClass) )
	{
		std::cout << "Register msgWnd class failed" << std::endl;
		return false;
	}
	
	msgWnd = CreateWindowEx(0, _T("Server Window Class"), _T("Server Titlebar"), WS_OVERLAPPEDWINDOW, CW_DEFAULT, CW_DEFAULT, CW_DEFAULT, CW_DEFAULT, HWND_MESSAGE, (HMENU)NULL, (HINSTANCE)NULL, (LPVOID)NULL);

	if (msgWnd == NULL) 
	{
		std::cout << "Failed to create window" << std::endl;
		return false;
	}

	SetWindowLongPtr(msgWnd, GWLP_USERDATA, (LONG)this);

	if(listen(tcpSocket.handle, 5) == SOCKET_ERROR)
	{
		std::cout << "Listen failed" << std::endl;
		return false;
	}

	if (WSAAsyncSelect(tcpSocket.handle, msgWnd, NOTIFICATION_NUMBER, (FD_ACCEPT | FD_READ | FD_CLOSE)) != 0)
	{
		printf("Failed to init async socket: (%d)[1]\n", WSAGetLastError());
		return false;
	}

	return true;
}

void NetworkManager::Update()
{
	if (clients.size() > 0)
	{
//		if (lastTimestepSent < volume->currentTimestep || lastTimestepSent == volume->timesteps-1)
//			SendState();
	}
}


void NetworkManager::SendState(int numXBlocks, int numYBlocks, int numZBlocks, int blockRes)
{
//	lastTimestepSent = volume->currentTimestep;

	// Then send entire volume
	for (int k=0; k<numZBlocks; k++)
		for (int j=0; j<numYBlocks; j++)
			for (int i=0; i<numXBlocks; i++)
			{
				SendBlock(i, j, k, blockRes);
			}
}


void NetworkManager::SendBlock(int i, int j, int k, int blockRes)
{
	int ID;

	int xMin = i * blockRes;
	int yMin = j * blockRes;
	int zMin = k * blockRes;

	Packet packet(PacketType::BLOCK);

	packet.WriteInt(blockRes);
	packet.WriteInt(i);
	packet.WriteInt(j);
	packet.WriteInt(k);

	for (int z=0; z<blockRes; z++)
		for (int y=0; y<blockRes; y++)
			for (int x=0; x<blockRes; x++)
			{
				if ((xMin + x) >= volume->xRes || (yMin + y) >= volume->yRes || (zMin + z) >= volume->zRes)
					continue;

				ID = (xMin + x) + ((yMin + y) * volume->xRes) + ((zMin + z) * volume->xRes * volume->yRes);

				packet.WriteByte(volume->currMemblock[ID]);
			}

	for (auto &client : clients)
	{
		if (!udpSocket.Send(client.ipAddress, packet))
			std::cout << "Failed to send " << packet.size << "bytes to " << client.ipAddress.udpPort << " - Error: " << WSAGetLastError() << std::endl;
		else
			std::cout << "Sent: " << i << " - " << j << " - " << k << std::endl;
	}
}



void NetworkManager::InitializeClient(Client &client)
{
	// Send initial info
	int ID;

	Packet initPacket;
	initPacket.WriteInt(0);
	initPacket.WriteByte((unsigned char)PacketType::INITIALIZATION);
	initPacket.WriteInt(volume->timesteps);
	initPacket.WriteFloat(volume->timePerFrame);
	initPacket.WriteInt(volume->xRes);
	initPacket.WriteInt(volume->yRes);
	initPacket.WriteInt(volume->zRes);
	initPacket.WriteInt(volume->bytesPerElement);
	initPacket.WriteBool(volume->littleEndian);

	initPacket.WriteCheckSum();

	bool initSent = false;

	while (!initSent)
		initSent = client.linkedSocket.Send(initPacket);


//	lastTimestepSent = volume->currentTimestep;
//
//	// Then send entire volume
//	for (int k=0; k<tempCoh->numZBlocks; k++)
//		for (int j=0; j<tempCoh->numYBlocks; j++)
//			for (int i=0; i<tempCoh->numXBlocks; i++)
//			{
//				int xMin = i * tempCoh->blockRes;
//				int yMin = j * tempCoh->blockRes;
//				int zMin = k * tempCoh->blockRes;
//
//				Packet packet;
//				packet.WriteInt(0);
//				packet.WriteByte((unsigned char)PacketType::BLOCK);
//
//				packet.WriteInt(i);
//				packet.WriteInt(j);
//				packet.WriteInt(k);
//
//				for (int z=0; z<tempCoh->blockRes; z++)
//					for (int y=0; y<tempCoh->blockRes; y++)
//						for (int x=0; x<tempCoh->blockRes; x++)
//						{
//							if ((xMin + x) >= volume->xRes || (yMin + y) >= volume->yRes || (zMin + z) >= volume->zRes)
//								continue;
//
//							ID = (xMin + x) + ((yMin + y) * volume->xRes) + ((zMin + z) * volume->xRes * volume->yRes);
//
//							packet.WriteByte(volume->currMemblock[ID]);
//						}
//
//				packet.WriteCheckSum();
//
//				bool sentSuccess = false;
//
//				while (!sentSuccess)
//					sentSuccess = client.linkedSocket.Send(packet);
//
//				std::cout << "Geometry packet sent size: " << packet.size << std::endl;
//			}
}





void NetworkManager::ReadMessage(WPARAM wParam)
{
	Packet packet;

	packet.size = recv(wParam, (char*)packet.message, MAX_PACKET_SIZE, 0);

	if (packet.size <= 0)
		return;

	packet.type = (PacketType)packet.ReadByte();

	switch (packet.type)
	{
		case PacketType::INITIALIZATION:
			for (auto &client : clients)
			{
				if (client.linkedSocket.handle == wParam)
				{
					client.ipAddress.udpPort = packet.ReadShort();

					InitializeClient(client);
				}
			}
			break;
	}	
}



LRESULT CALLBACK  NetworkManager::ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	sockaddr_in fromAddr;
	int fromLength = sizeof(fromAddr);
	IPAddress sender;

	switch (lParam)
	{
		case FD_ACCEPT:
		{
			std::cout << "Incoming connection" << std::endl;

			int tempSock = accept(tcpSocket.handle, (sockaddr*)&fromAddr, &fromLength);

			if (tempSock != INVALID_SOCKET)
			{
				sender.address = ntohl(fromAddr.sin_addr.s_addr);
				sender.tcpPort = ntohs(fromAddr.sin_port);
				std::cout << "Accepted connection from " << (int)sender.A() << "." << (int)sender.B() << "." << (int)sender.C() << "." << (int)sender.D() << ":" << sender.tcpPort << std::endl;

				clients.push_back(Client(sender, tempSock));
			}
			else
				std::cout << "Connection accept failed: " << WSAGetLastError() << std::endl;		

		    break;
		}

		case FD_READ:
		    std::cout << "Incoming data" << std::endl;
			ReadMessage(wParam);
		    break;
		
		case FD_CLOSE:
			for (auto it = clients.begin(); it != clients.end(); it++)
			{
				if (it->linkedSocket.handle == wParam)
				{
					clients.erase(it);
					break;
				}
			}

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