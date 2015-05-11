#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "UDPSocket.h"
#include "TCPSocket.h"
#include <vector>
#include <tchar.h>

#define IMAGE_BLOCK_RES 16

struct Client
{
	IPAddress ipAddress;
	TCPSocket linkedSocket;

	Client() { }

	Client(IPAddress addr, int sockHandle)
	{
		ipAddress = addr;
		linkedSocket.handle = sockHandle;
	}
};

class NetworkManager
{
public:
	std::vector<Client> clients;

	UDPSocket udpSocket;
	TCPSocket tcpSocket;
	HWND msgWnd;
	
	int udpPort, tcpPort;

	void Init();
	void Update(int screenWidth, int screenHeight, unsigned char *pixelBuffer);
	void SendState();
	void SendBlock(int i, int j, int screenWidth, int screenHeight, unsigned char *pixelBuffer);

	void InitializeClient(Client &client);

	bool InitTCPMsgHandler();
	void ReadMessage(WPARAM wParam);
	LRESULT CALLBACK  ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
	static LRESULT CALLBACK MsgWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
};


#endif